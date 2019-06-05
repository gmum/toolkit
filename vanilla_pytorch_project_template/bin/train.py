#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains simple CNN on cifar10/cifar100

Run like:
    * python bin/train.py cifar10 results/test_run
    * python bin/train.py cifar10 results/test_run --model.n_filters=20
    * python bin/train.py cifar10_lenet results/test_run
"""

import gin
from gin.config import _OPERATIVE_CONFIG

from src.data import get_dataset
from src import models
from src.training_loop import training_loop
from src.callbacks import get_callback
from src.utils import summary, acc, gin_wrap

import torch
from poutyne.framework import Model

import logging
logger = logging.getLogger(__name__)

@gin.configurable
def train(save_path, model, lr=0.1, batch_size=128, callbacks=[]):
    train, test, meta_data = get_dataset(batch_size=batch_size)

    pytorch_model_builder = models.__dict__[model]
    pytorch_model = pytorch_model_builder()
    summary(pytorch_model)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(pytorch_model.parameters(), lr=lr)
    model = Model(model=pytorch_model, optimizer=optimizer, loss_function=loss_function, metrics=[acc])

    callbacks_constructed = []
    for name in callbacks:
        clbk = get_callback(name, verbose=0)
        if clbk is not None:
            callbacks_constructed.append(clbk)

    steps_per_epoch = (len(meta_data['x_train']) - 1) // batch_size + 1
    training_loop(model=model, train=train, valid=test, save_path=save_path, config=_OPERATIVE_CONFIG,
                  use_tb=True, meta_data=meta_data,  steps_per_epoch=steps_per_epoch,
                  custom_callbacks=callbacks_constructed)


if __name__ == "__main__":
    gin_wrap(train)
