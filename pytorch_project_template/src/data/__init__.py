# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid.
"""
import os
import gin
from functools import partial
import logging
import numpy as np

from .datasets import cifar, mnist
from .streams import DatasetGenerator

logger = logging.getLogger(__name__)

@gin.configurable
def get_dataset(dataset, n_examples, data_seed, batch_size):
    train, valid, test, meta_data = globals()[dataset](seed=data_seed)

    if n_examples > 0:
        assert len(train[0]) >= n_examples
        train = [train[0][0:n_examples], train[1][0:n_examples]]

    # Configure stream + optionally augmentation
    meta_data['x_train'] = train[0]
    meta_data['y_train'] = train[1]

    train_stream = DatasetGenerator(train, seed=data_seed, batch_size=batch_size, shuffle=True)

    # Save some extra versions of the dataset. Just a pattern that is useful.
    train_stream_duplicated =  DatasetGenerator(train, seed=data_seed, batch_size=batch_size, shuffle=True)
    x_train_aug, y_train_aug = [], []
    n = 0
    for x, y in train_stream_duplicated:
        x_train_aug.append(x)
        y_train_aug.append(y)
        n += len(x)
        if n >= len(meta_data['x_train']):
            break
    meta_data['x_train_aug'] = np.concatenate(x_train_aug, axis=0)[0:len(meta_data['x_train'])]
    meta_data['y_train_aug'] = np.concatenate(y_train_aug, axis=0)[0:len(meta_data['x_train'])]
    meta_data['train_stream_duplicated'] = train_stream_duplicated

    # Return
    valid = DatasetGenerator(valid, seed=data_seed, batch_size=batch_size, shuffle=False)
    test = DatasetGenerator(test, seed=data_seed, batch_size=batch_size, shuffle=False)
    return train_stream, valid, test, meta_data
