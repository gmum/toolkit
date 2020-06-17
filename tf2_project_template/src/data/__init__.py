# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid.
"""
import logging
from .datasets import cifar

logger = logging.getLogger(__name__)

def get_dataset(dataset, seed, **kwargs):
    bundle = globals()[dataset](seed=seed, **kwargs)
    logger.info("Loaded dataset of name {} with x_train.shape={} and num_classes={}".format(dataset, bundle[-1]['input_shape'], bundle[-1]['num_classes']))
    return bundle
