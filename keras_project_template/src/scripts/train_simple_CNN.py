#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains simple CNN on cifar10/cifar100

Run like: python src/scripts/train_simple_CNN.py cifar10 results/test_run
"""

from keras.optimizers import SGD

from src.configs.simple_CNN import simple_CNN_configs
from src.data import cifar
from src.models import build_simple_model
from src.training_loop import cifar_training_loop
from src.vegab import main, MetaSaver, AutomaticNamer

def train(config, save_path):
    # Load data
    train, test, _ = cifar(dataset=config['dataset'], batch_size=config['batch_size'],
                           augmented=config['augmented'], preprocessing='center')

    # Load model

    model = build_simple_model(config)
    optimizer = SGD(lr=config['lr_schedule'][0][0], momentum=0.9)
    model.compile(optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Call training loop (warning: using test as valid. Please don't do this)
    cifar_training_loop(model=model,  train=train, valid=test, learning_rate_schedule=config['lr_schedule'],
        save_path=save_path, n_epochs=config['n_epochs'])


if __name__ == "__main__":
    main(simple_CNN_configs, train,
        plugins=[MetaSaver()])
