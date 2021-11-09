from einops import rearrange
from datamodules import *
from Audiomer import AudiomerClassification
from functools import partial
import pytorch_lightning as pl
from argparse import ArgumentParser
import numpy as np
import torch.nn.functional as F
import torch
from Metrics import *

def single_label_cross_entropy(y_pred, y):
    return F.nll_loss(y_pred, y)


class PartialModule(torch.nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def forward(self, x):
        return self.func(x, **self.kwargs)


class Experiment(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.model_params = dict(
            expansion_factor=2,
            mlp_dropout=0.2,
            num_heads=2,
            depth=1,
            dim_head=32,
            use_residual=kwargs['no_residual'],
            use_attention=kwargs['no_attention'],
            equal_strides=kwargs['unequal_strides'],
            use_se=kwargs['no_se'],
        )
        self.activation_function = torch.nn.Identity()

        self.model_params['input_size'] = 8192
        self.loss_fn = single_label_cross_entropy
        self.activation_function = PartialModule(torch.log_softmax, dim=-1)
        self.model_params['pool'] = 'cls'

        # networks
        if kwargs['dataset'] == 'SC35':
            self.model_params['num_classes'] = 35

        elif kwargs['dataset'] == 'SC12':
            self.model_params['num_classes'] = 12

        elif kwargs['dataset'] == 'SC20':
            self.model_params['num_classes'] = 20

        if kwargs['model'] == "L":
            self.model_params['config'] = [1, 4, 8, 16, 16, 32, 32, 64, 64, 96, 96, 192]
        elif kwargs['model'] == "M":
            self.model_params['config'] = [1, 4, 8, 16, 16, 32, 64, 128]
        elif kwargs['model'] == "S":
            self.model_params['config'] = [1, 4, 8, 8, 16, 16, 32, 32, 64, 64]

        self.model_params['kernel_sizes'] = [5] * \
            (len(self.model_params['config'])-1)

        self.model_params['mlp_dim'] = self.model_params['config'][-1]
        self.model = AudiomerClassification(**self.model_params)
        self.metrics = SpeechCommandsMetrics(
            self.model_params['num_classes'])

        # makes self.hparams under the hood and saves to ckpt
        for k, v in self.model_params.items():
            self.hparams[k] = v

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.activation_function(self(x))
        loss = self.loss_fn(y_pred, y)

        m = self.metrics(y_pred, y, "train")
        for metric_name, metric_value in m.items():
            self.log(metric_name+"/train", metric_value, prog_bar=True)
        self.log("LOSS/train", loss)

        return loss

    def training_epoch_end(self, *args, **kwargs):
        self.metrics.reset("train")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.activation_function(self(x))
        loss = self.loss_fn(y_pred, y)

        m = self.metrics(y_pred, y, "val")
        self.log("LOSS/val", loss, prog_bar=True)
        m['LOSS'] = loss
        return m

    def validation_epoch_end(self, outs):
        avg_loss = sum([item['LOSS'] for item in outs]) / len(outs)
        m = outs[-1]
        m['LOSS'] = avg_loss
        for metric_name, metric_value in m.items():
            self.log(metric_name+"/val", metric_value)
        self.metrics.reset("val")

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.activation_function(self(x))
        loss = self.loss_fn(y_pred, y)

        m = self.metrics(y_pred, y, "test")
        m['LOSS'] = loss
        return m

    def test_epoch_end(self, outs):
        avg_loss = sum([item['LOSS'] for item in outs]) / len(outs)
        m = outs[-1]
        m['LOSS'] = avg_loss
        for metric_name, metric_value in m.items():
            self.log(metric_name+"/test", metric_value)
        self.metrics.reset("test")
        return m

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt = torch.optim.Adam(self.model.parameters(),
                               lr=lr)

        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=320)

        return [opt], [schedule]

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--learning_rate", type=float, default=0.005, help="adam: learning rate"
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=2,
            help="Batch size",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=12,
            help="num_workers",
        )
        parser.add_argument(
            "--pin_memory",
            type=bool,
            default=False,
            help="Pin memory",
        )
        parser.add_argument(
            "--no_se",
            default=True,
            action='store_false',
            help="Whether to use squeeze excitation or not",
        )
        parser.add_argument(
            "--unequal_strides",
            default=False,
            action='store_true',
            help="Whether to use equal strides or not",
        )
        parser.add_argument(
            "--no_attention",
            default=True,
            action='store_false',
            help="Whether to use performer attention or not",
        )
        parser.add_argument(
            "--no_residual",
            default=True,
            action='store_false',
            help="Whether to use residual connections or not",
        )
        return parser

from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

def cli_main(args=None):

    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["SC35", "SC20", "SC12"],
                        help="SC35, SC12, SC20")
    parser.add_argument("--model", required=True, choices=["L", "M", "S"],
                        help="L, M, S")
    parser.add_argument("--checkpoint_path", required=True, type=str,
                        help="path to trained model")
    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "SC35":
        dm_cls = SpeechCommands35DataModule
    elif script_args.dataset == "SC12":
        dm_cls = SpeechCommands12DataModule
    elif script_args.dataset == "SC20":
        dm_cls = SpeechCommands20DataModule

    parser = dm_cls.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Experiment.add_model_specific_args(parser)
    args, _ = parser.parse_known_args(args)

    model = Experiment(**vars(args))
    model.load_state_dict(torch.load(script_args.checkpoint_path)['state_dict'])
    model.metrics.to("cuda")

    dm = dm_cls(batch_size=args.batch_size, num_workers=args.num_workers,
                pin_memory=args.pin_memory, augmentation=True)

    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.test(model, datamodule=dm) # dont train when evaluating


if __name__ == "__main__":
    cli_main()
