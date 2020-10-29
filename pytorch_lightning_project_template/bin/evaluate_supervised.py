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

import torch
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

from src.data import get_dataset
from src.utils import summary, acc, gin_wrap, parse_gin_config
from src.modules.supervised_training import SupervisedLearning
# Ensure gin seens all classes
from bin.train_supervised import *

import argh

def evaluate(save_path, checkpoint_name="weights.ckpt"):
    # Note: for resumed training you need to manually specify latest version

    # Load config
    config = parse_gin_config(os.path.join(save_path, "config.gin"))
    gin.parse_config_files_and_bindings([os.path.join(os.path.join(save_path, "config.gin"))], bindings=[""])

    # Create dynamically dataset generators
    train, valid, test, meta_data = get_dataset(batch_size=config['train.batch_size'], seed=config['train.seed'])

    ckpt_path = os.path.join(save_path, checkpoint_name)

    pl_module = SupervisedLearning.load_from_checkpoint(ckpt_path)

    trainer = pl.Trainer()
    results, = trainer.test(model=pl_module, test_dataloaders=test, ckpt_path=ckpt_path)
    logger.info(results)
    with open(os.path.join(save_path, "eval_results.json"), "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    argh.dispatch_command(evaluate)
