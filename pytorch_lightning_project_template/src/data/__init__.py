# -*- coding: utf-8 -*-
"""
Datasets available for use in the project.
"""
import logging
import gin

from .datasets import cifar, stl10

logger = logging.getLogger(__name__)

@gin.configurable
def get_dataset(dataset, seed, **kwargs):
    bundle = globals()[dataset](seed=seed, **kwargs)
    logger.info("Loaded dataset of name {} with x_train.shape={}".format(dataset, bundle[-1]['input_shape']))
    return bundle
