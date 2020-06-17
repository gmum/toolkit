#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer script for the project.

Example run commands:

* python bin/train.py tst configs/scnn.gin - Trains SimpleCNN on the CIFAR-10 dataset
"""

import gin
import os
import logging
import json
import tensorflow as tf
import numpy as np

from gin.config import _CONFIG
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import categorical_accuracy

from src.data import get_dataset
from src import models
from src.training_loop import training_loop, evaluate, restore_model
from src.callbacks import get_callback
from src.utils import gin_wrap

logger = logging.getLogger(__name__)

@gin.configurable
def train(save_path,
          model,
          datasets=['cifar10'],
          optimizer="SGD",
          data_seed=777,
          seed=777,
          batch_size=128,
          lr=0.0,
          wd=0.0,
          nesterov=False,
          checkpoint_monitor='val_categorical_accuracy:0',
          loss='ce',
          steps_per_epoch=-1,
          momentum=0.9,
          testing=False,
          testing_reload_best_val=True,
          callbacks=[]):
    np.random.seed(seed)

    # Create dataset generators (seeded)
    datasets = [get_dataset(d, seed=data_seed, batch_size=batch_size) for d in datasets]

    # Create model
    model = models.__dict__[model](input_shape=datasets[0][-1]['input_shape'], n_classes=datasets[0][-1]['num_classes'])
    logger.info("# of parameters " + str(sum([np.prod(p.shape) for p in model.trainable_weights])))
    model.summary()
    if loss == 'ce':
        loss_function = tf.keras.losses.categorical_crossentropy
    else:
        raise NotImplementedError()

    if optimizer == "SGD":
        optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=nesterov)
    elif optimizer == "Adam":
        optimizer = Adam(learning_rate=lr)
    else:
        raise NotImplementedError()

    # Create callbacks
    callbacks_constructed = []
    for name in callbacks:
        clbk = get_callback(name, verbose=0)
        if clbk is not None:
            callbacks_constructed.append(clbk)
        else:
            raise NotImplementedError(f"Did not find callback {name}")

    # Pass everything to the training loop
    metrics = [categorical_accuracy]

    if steps_per_epoch == -1:
        steps_per_epoch = (datasets[0][-1]['n_examples_train'] + batch_size - 1) // batch_size

    training_loop(model=model, optimizer=optimizer, loss_function=loss_function, metrics=metrics, datasets=datasets,
                  weight_decay=wd, save_path=save_path, config=_CONFIG, steps_per_epoch=steps_per_epoch,
                  use_tb=True, checkpoint_monitor=checkpoint_monitor, custom_callbacks=callbacks_constructed, seed=seed)

    if testing:
        if testing_reload_best_val:
            model = restore_model(model, os.path.join(save_path, "model_best_val.h5"))

        m_val = evaluate(model, [datasets[0][1]], loss_function, metrics)
        m_test = evaluate(model, [datasets[0][2]], loss_function, metrics)

        logger.info("Saving")
        eval_results = {}
        for k in m_test:
            eval_results['test_' + k] = float(m_test[k])
        for k in m_val:
            eval_results['val_' + k] = float(m_val[k])
        logger.info(eval_results)
        json.dump(eval_results, open(os.path.join(save_path, "eval_results.json"), "w"))


if __name__ == "__main__":
    gin_wrap(train)
