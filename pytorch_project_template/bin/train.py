#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains simple CNN on cifar10/cifar100

Run like: python bin/train.py cifar10 results/test_run
"""

from src.configs.simple_CNN import simple_CNN_configs
from src.data import get_cifar
from src.models import SimpleCNN
from src.training_loop import training_loop
from src.callbacks import LRSchedule
from src.vegab import wrap
from src.utils import summary, acc

import torch
from pytoune.framework import Model

import logging
logger = logging.getLogger(__name__)

def train(config, save_path):
    train, test, meta_data = get_cifar(dataset=config['dataset'], batch_size=config['batch_size'],
        augmented=config['augmented'], preprocessing='center', seed=config['seed'])

    pytorch_model = SimpleCNN(config)
    summary(pytorch_model)
    loss_function = torch.nn.MSELoss()  # Because logsoftmax. Be careful!
    optimizer = torch.optim.SGD(pytorch_model.parameters(), lr=config['lr'])
    model = Model(pytorch_model, optimizer, loss_function, [acc])

    callbacks = []
    callbacks.append(LRSchedule(lr_schedule=config['lr_schedule']))

    # Call training loop (warning: using test as valid. Please don't do this)
    steps_per_epoch = int(len(meta_data['x_train']) / config['batch_size'])
    training_loop(model=model,  train=train, valid=test, save_path=save_path, n_epochs=config['n_epochs'],
        save_freq=1, reload=config['reload'],
        steps_per_epoch=steps_per_epoch, custom_callbacks=callbacks)


if __name__ == "__main__":
    wrap(simple_CNN_configs, train)
