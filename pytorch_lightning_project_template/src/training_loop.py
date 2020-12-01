# -*- coding: utf-8 -*-
"""
Training loop based on PytorchLightning
"""

import logging
import os
import sys
import tqdm
import copy
import pickle
import numpy as np
import pandas as pd
import torch
import gin
import pytorch_lightning as pl

from functools import partial
from collections import defaultdict
from contextlib import contextmanager
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from src.utils import save_weights, parse_gin_config
from src.callbacks.base import MetaSaver, Heartbeat
from src import models, NEPTUNE_TOKEN, NEPTUNE_USER, NEPTUNE_PROJECT

logger = logging.getLogger(__name__)


@gin.configurable
def training_loop(train, valid, save_path, pl_module, callbacks,
        n_epochs, checkpoint_callback, use_neptune=False, resume=True, limit_train_batches=None, neptune_tags="", neptune_name=""):
    """
    Largely model/application agnostic training code.
    """
    # Train with proper resuming
    # Copy gin configs used, for reference, to the save folder

    if not limit_train_batches:
        limit_train_batches = len(train)
        
    os.system("rm " + os.path.join(save_path, "*gin"))
    for gin_config in sys.argv[2].split(";"):
        os.system("cp {} {}/base_config.gin".format(gin_config, save_path))
    with open(os.path.join(save_path, "config.gin"), "w") as f:
        f.write(gin.operative_config_str())
    hparams = parse_gin_config(os.path.join(save_path, 'config.gin'))
    if 'train.callbacks' in hparams:
        del hparams['train.callbacks']
    # TODO: What is a less messy way to pass hparams? This is only that logging is aware of hyperparameters
    pl_module._set_hparams(hparams)
    pl_module._hparams_initial = copy.deepcopy(hparams)
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
        callbacks += [MetaSaver(), Heartbeat(), LearningRateMonitor()]
    trainer = pl.Trainer(
        default_root_dir=save_path,
        limit_train_batches=limit_train_batches,
        max_epochs=n_epochs,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=os.path.join(save_path, 'last.ckpt')
        if resume and os.path.exists(os.path.join(save_path, 'last.ckpt')) else None)
    trainer.fit(pl_module, train, valid)
    return trainer
