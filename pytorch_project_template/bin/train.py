#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains simple CNN on cifar10/cifar100

Run like: python bin/train.py cifar10 results/test_run
Reload like python bin/train.py cifar10 results/test_run --reload
Pass LR schedule like: python bin/train.py cifar10 results/test_run --lr_schedule=[[1]]
or python bin/train.py cifar10 results/test_run --lr_schedule="[[1]]"
"""

from src.configs.simple_CNN import simple_CNN_configs
from src.data import get_cifar
from src.models import SimpleCNN
from src.training_loop import training_loop
from src.vegab import wrap
from src.utils import summary

import torch
from pytoune.framework import Model

def acc(y_pred, y_true):
    _, y_pred = y_pred.max(1)
    _, y_true = y_true.max(1)
    acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100

def train(config, save_path):
    train, test, meta_data = get_cifar(dataset=config['dataset'], batch_size=config['batch_size'],
        augmented=config['augmented'], preprocessing='center', seed=config['seed'])

    pytorch_model = SimpleCNN(config)
    summary(pytorch_model)
    loss_function = torch.nn.MSELoss()  # Because logsoftmax. Be careful!
    # loss_function = torch.nn.CrossEntropyLoss()  # Because logsoftmax. Be careful!
    optimizer = torch.optim.SGD(pytorch_model.parameters(), lr=config['lr'])
    model = Model(pytorch_model, optimizer, loss_function, [acc])

    # TODO: Fix
    # def lr_schedule(epoch, logs):
    #     for e, v in learning_rate_schedule:
    #         if epoch >= e:
    #             model.optimizer.lr.set_value(v)
    #             break
    #     logger.info("Fix learning rate to {}".format(v))
    #
    # callbacks.append(LambdaCallback(on_epoch_end=lr_schedule))

    # training_loop(model=model, train=train, steps_per_epoch=steps_per_epoch, save_freq=config['save_freq'],
    #     checkpoint_monitor="val_acc", epochs=config['n_epochs'], save_path=save_path,
    #     reload=config['reload'],
    #     valid=valid, custom_callbacks=callbacks, verbose=2)

    # Call training loop (warning: using test as valid. Please don't do this)
    steps_per_epoch = int(len(meta_data['x_train']) / config['batch_size'])
    training_loop(model=model,  train=train, valid=test, save_path=save_path, n_epochs=config['n_epochs'],
        save_freq=1, reload=config['reload'],
        steps_per_epoch=steps_per_epoch)


if __name__ == "__main__":
    wrap(simple_CNN_configs, train)
