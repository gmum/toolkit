#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer script for src.pl_modules.supervised_learning. Example run command: bin/train.py save_to_folder configs/cnn.gin.
"""

import gin
from gin.config import _CONFIG
import torch
import logging
import os
import json
import time
import datetime
import sys
import copy

import torch
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

from src.data import get_dataset
from src import models, NEPTUNE_TOKEN, NEPTUNE_USER, NEPTUNE_PROJECT
# from src.training_loop import training_loop
from src.callbacks import get_callback
from src.callbacks.base import MetaSaver, Heartbeat
from src.utils import summary, acc, gin_wrap, parse_gin_config
from src.modules import supervised_training

from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers


@gin.configurable
def train(save_path, model, batch_size=128, seed=777, n_epochs=2,
        callbacks=[], resume=True, evaluate=True, limit_train_batches=2,
        use_neptune=False, neptune_tags="", neptune_name=""):
    # Create dynamically dataset generators
    train, valid, test, meta_data = get_dataset(batch_size=batch_size, seed=seed)

    # Create dynamically model
    model = models.__dict__[model]()
    summary(model)

    # Create dynamically callbacks
    callbacks_constructed = []
    for name in callbacks:
        clbk = get_callback(name, verbose=0)
        if clbk is not None:
            callbacks_constructed.append(clbk)

    if not resume and os.path.exists(os.path.join(save_path, "last.ckpt")):
        raise IOError("Please clear folder before running or pass train.resume=True")

    # Train with proper resuming
    # Copy gin configs used, for reference, to the save folder
    os.system("rm " + os.path.join(save_path, "*gin"))
    for gin_config in sys.argv[2].split(";"):
        os.system("cp {} {}/base_config.gin".format(gin_config, save_path))
    with open(os.path.join(save_path, "config.gin"), "w") as f:
        f.write(gin.operative_config_str())
    hparams = parse_gin_config(os.path.join(save_path, 'config.gin'))
    if 'train.callbacks' in hparams:
        del hparams['train.callbacks']
    pl_module = supervised_training.SupervisedLearning(model)
    # TODO: What is a less messy way to pass hparams? This is only that logging is aware of hyperparameters
    pl_module._set_hparams(hparams)
    pl_module._hparams_initial = copy.deepcopy(hparams)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(save_path, "weights"),
        verbose=True,
        save_last=True,  # For resumability
        monitor='valid_acc',
        mode='max'
    )
    loggers = []
    loggers.append(pl_loggers.CSVLogger(save_path))
    if use_neptune:
        from pytorch_lightning.loggers import NeptuneLogger
        loggers.append(NeptuneLogger(
            api_key=NEPTUNE_TOKEN,
            project_name=NEPTUNE_USER + "/" + NEPTUNE_PROJECT,
            experiment_name=neptune_name if len(neptune_name) else os.path.basename(save_path),
            tags=neptune_tags.split(',') if len(neptune_tags) else None,
        ))
    callbacks_constructed += [MetaSaver(), Heartbeat(), LearningRateMonitor()]
    trainer = pl.Trainer(
        default_root_dir=save_path,
        limit_train_batches=limit_train_batches,
        max_epochs=n_epochs,
        logger=loggers,
        callbacks=callbacks_constructed,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=os.path.join(save_path, 'last.ckpt')
        if resume and os.path.exists(os.path.join(save_path, 'last.ckpt')) else None)
    trainer.fit(pl_module, train, valid)

    # Evaluate
    if evaluate:
        results, = trainer.test(test_dataloaders=test)
        logger.info(results)
        with open(os.path.join(save_path, "eval_results.json"), "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    gin_wrap(train)
