from argparse import ArgumentParser

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import Cifar100Model
from dataset import DatasetModule

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    args = ArgumentParser()
    args.add_argument('--batch-size', default=32, type=int)
    args.add_argument('--epochs', default=50, type=int)
    args.add_argument('--wandb-project', default='mlops_finetune_cifar100', type=str)

    args = args.parse_args()

    pl.seed_everything(43)
    wandb_logger = WandbLogger(project=args['wandb_project'], job_type='train')

    dm = DatasetModule(
        batch_size=args['batch_size']
    )
    model = Cifar100Model()

    run_name = wandb_logger.experiment.name
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/' + run_name,
        filename='{epoch:02d}--{val_acc:.2f}',
        save_top_k=2,
        mode="max",
        save_weights_only=True)
    early_stop_callback = EarlyStopping(monitor='val_acc', patience=5, verbose=True, mode='max')

    AVAIL_GPUS = min(1, torch.cuda.device_count())

    trainer = pl.Trainer(max_epochs=args['epochs'],
                         gpus=AVAIL_GPUS,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(model, dm)

    trainer.test(model, datamodule=dm)

    wandb.finish()
