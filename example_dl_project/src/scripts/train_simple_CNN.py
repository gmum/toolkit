#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains simple CNN on cifar10/cifar100
"""

from keras.optimizers import SGD

from src.configs.simple_CNN import simple_CNN_configs
from src.data import get_cifar
from src.models import build_simple_model
from src.training_loop import cifar_training_loop
from vegab import main, MetaSaver, AutomaticNamer


def train(config, save_path):
    # Load data
    train, test = get_cifar(dataset=config['dataset'], batch_size=config['batch_size'],
        augmented=config['augmented'], preprocessing='center')

    # Load model
    model = build_simple_model(config)

    # Optimizer
    optimizer = SGD(lr=config['learning_rate_schedule'])

    # Call training loop (warning: using test as valid. Please don't do this)
    cifar_training_loop(model=model, optimizer=optimizer,
        train=train, valid=test, learning_rate_schedule=config['learning_rate_schedule'],
        save_path=save_path, n_epochs=config['n_epochs'])


if __name__ == "__main__":
    main(simple_CNN_configs, train,
        plugins=[MetaSaver(), AutomaticNamer(as_prefix=True, namer="timestamp_namer")])
