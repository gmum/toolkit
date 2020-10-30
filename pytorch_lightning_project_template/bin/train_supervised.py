#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer script for src.pl_modules.supervised_learning. Example run command: bin/train.py save_to_folder configs/cnn.gin.
"""

import gin
import logging
import os
import json

from src.data import get_dataset
from src.callbacks import get_callback
from src.utils import summary, acc, gin_wrap, parse_gin_config
from src.modules import supervised_training
from src import models
from src.training_loop import training_loop

from pytorch_lightning.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)

@gin.configurable
def train(save_path, model, batch_size=128, seed=777, callbacks=[], resume=True, evaluate=True):
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

    # Create module and pass to trianing
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(save_path, "weights"),
        verbose=True,
        save_last=True,  # For resumability
        monitor='valid_acc',
        mode='max'
    )
    pl_module = supervised_training.SupervisedLearning(model, meta_data=meta_data)
    trainer = training_loop(train, valid, pl_module=pl_module, checkpoint_callback=checkpoint_callback,
        callbacks=callbacks_constructed, save_path=save_path)
    
    # Evaluate
    if evaluate:
        results, = trainer.test(test_dataloaders=test)
        logger.info(results)
        with open(os.path.join(save_path, "eval_results.json"), "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    gin_wrap(train)
