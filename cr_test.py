import cr_pl
from cr_pl import *
from vit_pytorch import ViT
import WideResNet
from WideResNet import *
from robustbench.data import load_cifar10
from autoattack import AutoAttack

# test model
def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

# pytorch moduel for ensembling models
class EnsembledModels(nn.Module):
    def __init__(self, models):
      super(EnsembledModels, self).__init__()
      self.model_lst = models

    def forward(self, x):
      logits = None
      for idx, model in enumerate(self.model_lst):
        if idx == 0:
          logits = model(x).cuda()
        else:
          logits += model(x).cuda()
      logits /= len(self.model_lst)
      return logits

def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    backbone_lst = []
    if hparams.backbone_model == 'WResNet':
        load_path_lst = hparams.load_path.split(', ')
        no_sn_lst = hparams.no_sn.split(', ')
        for idx, _ in enumerate(load_path_lst):
            if str(idx) in no_sn_lst:
                use_sn = False
            else:
                use_sn = hparams.use_sn
            backbone = WideResNet(depth=hparams.net_depth, 
                                n_classes=hparams.num_classes, 
                                widen_factor=hparams.wide_factor,
                                use_sn=use_sn)
            backbone_lst.append(backbone)
    else:
        raise NotImplementedError()
    #backbone = create_model()
    if hparams.num_examples != -1:
        x_test, y_test = load_cifar10(n_examples=hparams.num_examples)
    else:
        x_test, y_test = load_cifar10()
    print("Test Mode")
    if hparams.ensemble:
        model_lst = []
        for idx, load_path in enumerate(load_path_lst):
            model = CR_pl(hparams, backbone_lst[idx])
            checkpoint = torch.load(load_path)
            state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
            model.model.load_state_dict(state_dict, strict=True)
            backbone = model.model.cuda().eval()
            model_lst.append(model)
        model = EnsembledModels(model_lst)
    else:
        model = CR_pl(hparams, backbone_lst[0])
        # Test the model from loaded checkpoint
        checkpoint = torch.load(load_path_lst[0])
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
        model.model.load_state_dict(state_dict, strict=True)
        backbone = model.model.cuda()
        backbone.eval()

    adversary = AutoAttack(model, norm='Linf', eps=8/255)
    #adversary = AutoAttack(backbone, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test.cuda(), y_test.cuda())

if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = os.path.dirname('./cr_pl')
    parent_parser = argparse.ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='how many gpus'
    )

    parent_parser.add_argument(
        '--num_nodes',
        type=int,
        default=1,
        help='how many nodes'
    )

    parent_parser.add_argument(
        '--precision',
        type=int,
        default=16,
        help='default to use mixed precision 16'
    )

    parent_parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help='number of classes for the dataset'
    )

    parent_parser.add_argument(
        '--backbone_model',
        choices=["WResNet", "ViT"],
        type=str,
        default="WResNet",
        help='backbone model'
    )

    parent_parser.add_argument(
        '--net_depth',
        type=int,
        default=34,
        help='wide resnet depth'
    )

    parent_parser.add_argument(
        '--wide_factor',
        type=int,
        default=10,
        help='wide resenet wide factor'
    )

    parent_parser.add_argument(
        '--load_path',
        type=str,
        default=None,
        help='load path for saved models, separate by ","'
    )

    parent_parser.add_argument(
        '--no_sn',
        type=str,
        default=None,
        help='indices for loaded model that does not use spectral normalization'
    )

    parent_parser.add_argument(
        '--num_examples',
        type=int,
        default=-1,
        help='number of examples to generate for autoattack evaluation'
    )

    parent_parser.add_argument(
        '--use_sn', 
        action='store_true',
        help="Use Spectral Normalization",
    )

    parent_parser.add_argument(
        '--use_awp', 
        action='store_true', 
        help="Use Stochastic Weight Perturbation",
    )

    parent_parser.add_argument(
        '--use_swa', 
        action='store_true',
        help="Use Stochastic Weighted Average",
    )

    parent_parser.add_argument(
        '--ensemble',
        action='store_true',
        help='decide if ensembling the models, if not, load_path1 is used for model loading'
    )

    # each LightningModule defines arguments relevant to it
    parser = CR_pl.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)

