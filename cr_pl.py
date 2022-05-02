import argparse
from autoaugment import *
from dataset_utils import *
from attack_utils import *
from loss_utils import off_diagonal, _jensen_shannon_div, _H_min_div 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from transformers import get_cosine_schedule_with_warmup
import copy
import awp

class CR_pl(LightningModule):
  def __init__(self, hparams, backbone):
    super().__init__()

    self.args = hparams
    self.hparams.update(vars(hparams))
    
    # setup model
    self.model = backbone
    if hparams.extra_reg != None:
        self.projector = self.Projector(hparams.embed_dim)
    
    # setup criterion
    self.criterion = nn.CrossEntropyLoss()
    
    # use awp or not
    if hparams.use_awp:
      self.proxy = copy.deepcopy(self.model)
      proxy_opt = torch.optim.SGD(self.proxy.parameters(), lr=0.01)
      self.awp_adversary = awp.AdvWeightPerturb(proxy=self.proxy,
                                                proxy_optim=proxy_opt, 
                                                gamma=1e-2)

    # setup dataset and adversary attack
    self.kwargs = {'pin_memory': hparams.pin_memory, 'num_workers': hparams.num_workers}
    self.train_set, self.test_set, self.image_size, self.n_classes = get_dataset('autoaug', True)
    self.adversary = attack_module(self.model, self.criterion)

  def Projector(self, embedding):
    mlp_spec = f"{embedding}-{self.hparams.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.model(x)
    return out

  def _forward(self, batch, stage):
    images, labels = batch
    images_aug1, images_aug2 = images[0], images[1]
    images_pair = torch.cat([images_aug1, images_aug2], dim=0)  # 2B
    if stage == "train":
        images_adv = self.adversary(images_pair, labels.repeat(2))
        if self.hparams.use_awp:
            self.awp = self.awp_adversary.calc_awp(model=self.model,
                                                inputs_adv=images_adv,
                                                targets=labels.repeat(2))
            self.awp_adversary.perturb(self.model, self.awp)
    else:
        with torch.enable_grad():
            images_adv = self.adversary(images_pair, labels.repeat(2))

    # register hook to get intermediate output
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook
    # get original output
    outputs_adv = self.model(images_adv)

    if self.hparams.extra_reg == 'cov': 
        self.model.relu.register_forward_hook(get_activation('penultimate'))
        # get latent
        outputs_latent = activation['penultimate']
        outputs_latent = F.avg_pool2d(outputs_latent, 8)
        outputs_latent = outputs_latent.view(outputs_latent.size(0), -1)
    loss_ce = self.criterion(outputs_adv, labels.repeat(2))

    ### consistency regularization ###
    outputs_adv1, outputs_adv2 = outputs_adv.chunk(2)
    if self.hparams.loss_func == 'JS':
        loss_con = self.hparams.con_coeff * _jensen_shannon_div(outputs_adv1, outputs_adv2, self.hparams.T)
    elif self.hparams.loss_func == 'MIN':
        loss_con = self.hparams.con_coeff * _H_min_div(outputs_adv1, outputs_adv2, self.hparams.T) 
    else:
        raise NotImplementedError() 

    # calculate covariance regularization by projecting laten
    # representation with a given projector
    if self.hparams.extra_reg == 'cov':
        outputs_latent1, outputs_latent2 = outputs_latent.chunk(2)
        outputs_latent1, outputs_latent2 = self.projector(outputs_latent1), self.projector(outputs_latent2)
        cov_1 = (outputs_latent1.T @ outputs_latent1) / (self.hparams.batch_size - 1)
        cov_2 = (outputs_latent2.T @ outputs_latent2) / (self.hparams.batch_size - 1)
        loss_cov = off_diagonal(cov_1).pow_(2).sum().div(num_features) + \
                off_diagonal(cov_2).pow_(2).sum().div(num_features)
        loss_cov *= self.hparams.cov_coeff
        if stage:
            self.log(f"{stage}_loss_cov", loss_cov, prog_bar=True)
    else:
        loss_cov = 0
    ### total loss ###
    loss_ce *= self.hparams.sim_coeff
    loss = loss_ce + loss_con + loss_cov

    if stage:
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_loss_con", loss_con, prog_bar=True)
        self.log(f"{stage}_loss_sim", loss_ce, prog_bar=True)

    return loss

  def training_step(self, batch, batch_idx):
    loss = self._forward(batch, "train")
    return loss

  def training_step_end(self, batch_parts):
    if self.hparams.use_awp:
        self.awp_adversary.restore(self.model, self.awp)

  def validation_step(self, batch, batch_idx):
    self._forward(batch, "val")

  def test_step(self, batch, batch_idx):
    self._forward(batch, "test")

  def train_dataloader(self):
    trainloader = DataLoader(self.train_set, shuffle=True, batch_size=self.hparams.batch_size, **self.kwargs)
    self.train_len = len(trainloader)
    return trainloader

  def val_dataloader(self):
    valloader = DataLoader(self.test_set, shuffle=False, batch_size=self.hparams.batch_size, **self.kwargs)
    return valloader

  def test_dataloader(self):
    testloader = DataLoader(self.test_set, shuffle=False, batch_size=self.hparams.batch_size, **self.kwargs)
    return testloader

  def configure_optimizers(self):
    if self.hparams.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=5e-4,
        )
    elif self.hparams.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
    else:
        raise NotImplementedError()
    
    tb_size = self.hparams.batch_size * max(1, self.trainer.num_devices)
    ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
    self.total_steps = (len(self.train_set) // tb_size) // ab_size
    self.warmup_steps = 0.06 * self.total_steps
    
    if self.hparams.scheduler=="multistep":
        lr_decay_gamma = 0.1
        #milestones = [80, 100]
        milestones = [int(0.5 * self.hparams.max_epochs), int(0.75 * self.hparams.max_epochs)]
        scheduler = MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
    elif self.hparams.scheduler=="cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.warmup_steps,
                                                    num_training_steps=self.total_steps,
                                                )
    else:
        raise NotImplementedError()
    
    return {"optimizer": optimizer, "lr_scheduler": scheduler}

  def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
    """
    Parameters you define here will be available to your model through self.hparams
    :param parent_parser:
    :param root_dir:
    :return:
    """
    parser = argparse.ArgumentParser(parents=[parent_parser])

    # Model parameters
    parser.add_argument('--loss_func', choices=["JS", "MIN"], default="JS", type=str, help='divergence loss function for jensen s')
    parser.add_argument('--sim_coeff', default=1.0, type=float, help='scaling coefficient for supervised loss') 
    parser.add_argument('--con_coeff', default=1.0, type=float, help='scaling coefficient for consistency regularization')
    parser.add_argument('--cov_coeff', default=1.0, type=float, help='scaling coefficient for covariance regularization')
    parser.add_argument('--T', default=0.5, type=float, help='temperature hyperparameter for ')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate for optimizer')
    parser.add_argument('--embed_dim', default=640, type=int, help='embedding dimensionality for network output')
    parser.add_argument("--mlp", default="2048-2048-2048", type=str, help='Size and number of layers of the MLP expander head')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size for dataloader')
    parser.add_argument("--max_epochs", type=int, default=200, help='max epochs for training')
    parser.add_argument("--extra_reg", choices=["cov"], type=str, default=None, help='decide whether to use extra regularization')
    #parser.add_argument("--use_mixup", type=bool, default=False)
    parser.add_argument('--optimizer', choices=["adamw", "sgd"], default="sgd", type=str, help='optimizer choices')
    parser.add_argument('--scheduler', choices=["cosine", "multistep"], default="multistep", type=str, help='scheduler choices')
    parser.add_argument('--warmup', default=False, type=bool, help='decide whether to use warmup')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
    parser.add_argument('--pin_memory', default=True, type=bool, help='decide whether to use pin memory')
    return parser
