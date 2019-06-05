#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains simple CNN on cifar10/cifar100

Run like:
    * python bin/train.py cifar10 results/test_run
    * python bin/train.py cifar10 results/test_run --model.n_filters=20
    * python bin/train.py cifar10_lenet results/test_run
"""

from src.configs import cifar_train_configs
from src.data import datasets
from src import models
from src.training_loop import training_loop
from src.callbacks.callbacks import LRSchedule
from src.vegab import wrap
from src.utils import summary, acc

import torch
from pytoune.framework import Model

import logging
logger = logging.getLogger(__name__)

def train(config, save_path):
    train, test, meta_data = datasets(dataset=config['dataset'], batch_size=config['batch_size'],
                                      augmented=config['augmented'], preprocessing='center', seed=config['seed'])

    pytorch_model_builder = models.__dict__[config['model']]
    pytorch_model = pytorch_model_builder(**config.get('model_kwargs', {}))
    summary(pytorch_model)
    loss_function = torch.nn.MSELoss()  # Because logsoftmax. Be careful!
    optimizer = torch.optim.SGD(pytorch_model.parameters(), lr=config['lr'])
    model = Model(pytorch_model, optimizer, loss_function, [acc])

    callbacks = []
    callbacks.append(LRSchedule(lr_schedule=config['lr_schedule']))

    # Call training loop (warning: using test as valid. Please don't do this)
    steps_per_epoch = int(len(meta_data['x_train']) / config['batch_size'])
    training_loop(model=model,  train=train, valid=test, save_path=save_path, n_epochs=config['n_epochs'],
        save_freq=1, reload=config['reload'], use_tb=True,
        steps_per_epoch=steps_per_epoch, custom_callbacks=callbacks)


if __name__ == "__main__":
    wrap(cifar_train_configs, train)
