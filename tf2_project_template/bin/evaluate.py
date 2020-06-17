#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for the project. Example run command: bin/evaluate_tf.py path_to_experiment
"""

import gin
import logging
import sys
import json
import os

import tensorflow as tf
from tensorflow.keras.metrics import categorical_accuracy

logger = logging.getLogger(__name__)

from bin.train import *
from src.utils import restore_model
from src.plotting import load_HC
from src.training_loop import evaluate


if __name__ == "__main__":
    E = sys.argv[1]
    if len(sys.argv) == 3:
        checkpoint = sys.argv[2]
        suffix = checkpoint
    else:
        checkpoint = 'model_best_val.h5'
        suffix = ""

    H, C = load_HC(E)
    # Note: Load_HC doesn't load gin properly
    gin.parse_config_files_and_bindings([os.path.join(os.path.join(E, "config.gin"))], bindings=[""])
    logger.info(C['train']['datasets'])
    for dm in C['train']['datasets_modifiers']:
        dm['one_hot'] = True
    datasets = [get_dataset(d, seed=C['train']['data_seed'], batch_size=C['train']['batch_size'], **dm) for d, dm in
                zip(C['train']['datasets'], C['train']['datasets_modifiers'])]

    model = models.__dict__[C['train']['model']](input_shape=datasets[0][-1]['input_shape'],
                                                 n_classes=datasets[0][-1]['num_classes'])
    logger.info("# of parameters " + str(sum([np.prod(p.shape) for p in model.trainable_weights])))
    model.summary()

    if C['train']['loss'] == 'ce':
        loss_function = tf.keras.losses.categorical_crossentropy
    else:
        raise NotImplementedError()
    metrics = [categorical_accuracy]
    if C['train'].get("f1", False):
        metrics.append("f1")

    model = restore_model(model, os.path.join(E, checkpoint))

    m_test = evaluate(model, [datasets[0][2]], loss_function, metrics)
    m_val = evaluate(model, [datasets[0][1]], loss_function, metrics)

    logger.info("Saving")
    eval_results = {}
    for k in m_test:
        eval_results['test_' + k] = float(m_test[k])
    for k in m_val:
        eval_results['val_' + k] = float(m_val[k])
    logger.info(eval_results)

    json.dump(eval_results, open(os.path.join(E, f"eval_results{suffix}.json"), "w"))