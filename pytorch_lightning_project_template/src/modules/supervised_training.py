"""
A PyTorch Lighting module defines a *system* that is constructed from modules (data loaders, models, optimizer)
that trains and evaluates *on some task*.

The benefit is that such module can be then easily trained using various PL modules.

The most standard task is supervised training.
"""
import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule

import gin

@gin.configurable
class SupervisedLearning(LightningModule):
    """
    Module defining a supervised learning system.
    """

    def __init__(self, model, meta_data, lr=0.1):
        super().__init__()
        self.model = model
        self.meta_data = meta_data
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('train_loss', loss_val, on_epoch=True, on_step=True)
        self.log('train_acc', acc1, on_epoch=True, on_step=True)
        self.log('train_loss_step', loss_val, on_epoch=False, on_step=True)
        self.log('train_acc_step', acc1, on_epoch=False, on_step=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('valid_loss', loss_val, on_epoch=True)
        self.log('valid_acc', acc1, on_epoch=True)

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        # TODO: I am not sure where and how pytorch lightning uses it. Maybe not necessary to havet his logic here
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return optimizer

