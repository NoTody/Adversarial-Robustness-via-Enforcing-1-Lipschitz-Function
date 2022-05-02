import os
import WideResNet
from WideResNet import *
from cr_pl import *
from vit_pytorch import ViT
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging, TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy

SEED = 2022
seed_everything(SEED)

def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    if hparams.backbone_model == 'WResNet':
        backbone = WideResNet(depth=hparams.net_depth, 
                            n_classes=hparams.num_classes, 
                            widen_factor=hparams.wide_factor,
                            use_sn=hparams.use_sn)
    elif hparams.backbone_model == 'ViT':
        backbone = ViT(image_size=32, patch_size=4, num_classes=hparams.num_classes, dim=768, depth=12,
                    heads=16, mlp_dim=1024, dropout=0.1, emb_dropout=0.1, pool='mean')
    else:
        raise NotImplementedError()
    #backbone = create_model()
    cr_pl = CR_pl(hparams, backbone)
    # ------------------------
    # 2 DEFINE CALLBACKS
    # ------------------------
    bar = TQDMProgressBar(refresh_rate=1, process_position=0)
    if hparams.use_swa:
        swa_callback = StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=None, annealing_epochs=10, 
                                                annealing_strategy='cos', avg_fn=None)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(hparams.runpath, hparams.model_dir),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_weights_only=False
        )
    #early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    callbacks = [checkpoint_callback, bar]
    if hparams.use_swa:
        callbacks.append(swa_callback)
    trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        gpus=hparams.gpus,
        precision=hparams.precision,
        num_nodes=hparams.num_nodes,
        max_epochs=hparams.max_epochs,
        #strategy=None,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=callbacks
        )

    # ------------------------
    # 4 START TRAINING
    # ------------------------
    trainer.fit(cr_pl)
    if hparams.use_swa:
        path = os.path.join(hparams.runpath, hparams.model_dir, "swa.ckpt")
        trainer.save_checkpoint(path)

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
        choices=["ViT", "WResNet"],
        default="WResNet", 
        type=str,
        help='name of backbone model'
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
        '--runpath', 
        type=str, 
        default="./runs", 
        help = "the path to store all models"
    )

    parent_parser.add_argument(
        '--model_dir', 
        type=str,
        default="e1",
        help = "the path to store model directory"
    )

    # each LightningModule defines arguments relevant to it
    parser = CR_pl.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
