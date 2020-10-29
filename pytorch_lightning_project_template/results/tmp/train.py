#!/usr/bin/env python3
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
from src.callbacks import get_callback, LambdaCallbackPickableEveryKExamples
from src.vegab import wrap
from src.utils import summary, acc, run_with_redirection

import torch
from poutyne.framework import Model

import os
from functools import partial

import logging
logger = logging.getLogger(__name__)


def _train(config, save_path):
    train, test, meta_data = datasets(dataset=config['dataset'], batch_size=config['batch_size'],
                                      augmented=config['augmented'], preprocessing='center', seed=config['seed'], n_examples=config['n_examples'])

    pytorch_model_builder = models.__dict__[config['model']]
    pytorch_model = pytorch_model_builder(**config.get('model_kwargs', {}))
    summary(pytorch_model)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(pytorch_model.parameters(), lr=config['lr'])
    model = Model(model=pytorch_model, optimizer=optimizer, loss_function=loss_function, metrics=[acc])

    callbacks = []
    for k in config:
        clbk = get_callback(k, verbose=0, **config.get(k + "_kwargs", {}))
        if clbk is not None:
            callbacks.append(clbk)

    steps_per_epoch = (len(meta_data['x_train']) - 1) // config['batch_size'] + 1
    training_loop(model=model,  train=train, valid=test, save_path=save_path,
        use_tb=True, meta_data=meta_data, config=config,
        steps_per_epoch=steps_per_epoch, custom_callbacks=callbacks)

def train(config, save_path):
    run_with_redirection(
        os.path.join(save_path, 'stdout_my.txt'),
        os.path.join(save_path, 'stderr_my.txt'),
        partial(_train, config=config, save_path=save_path))()

if __name__ == "__main__":
    wrap(cifar_train_configs, train)
