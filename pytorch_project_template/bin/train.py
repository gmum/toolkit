#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer script. Example run command: bin/train.py save_to_folder configs/cnn.gin.
"""

import gin
from gin.config import _CONFIG
import torch
import logging
logger = logging.getLogger(__name__)

from src.data import get_dataset
from src import models
from src.training_loop import training_loop
from src.callbacks import get_callback
from src.utils import summary, acc, gin_wrap

@gin.configurable
def train(save_path, model, lr=0.1, batch_size=128, callbacks=[]):
    # Create dynamically dataset generators
    train, valid, test, meta_data = get_dataset(batch_size=batch_size)

    # Create dynamically model
    model = models.__dict__[model]()
    summary(model)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Create dynamically callbacks
    callbacks_constructed = []
    for name in callbacks:
        clbk = get_callback(name, verbose=0)
        if clbk is not None:
            callbacks_constructed.append(clbk)

    # Pass everything to the training loop
    training_loop(model=model, optimizer=optimizer, loss_function=loss_function, metrics=[acc],
                  train=train, valid=test, meta_data=meta_data, save_path=save_path, config=_CONFIG,
                  use_tb=True, custom_callbacks=callbacks_constructed)


if __name__ == "__main__":
    gin_wrap(train)
