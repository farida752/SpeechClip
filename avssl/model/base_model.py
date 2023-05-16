import abc

import pytorch_lightning as pl
import torch
from torch import nn, optim

from ..base import OrderedNamespace


class BaseLightningModel(pl.LightningModule):
    def __init__(self, config: OrderedNamespace):
        print("Hello from **model/base_model** class BaseLightningModel")
        super().__init__()
        self.config = config
        self.save_hyperparameters()

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abc.abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError
