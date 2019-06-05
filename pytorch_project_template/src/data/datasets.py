# -*- coding: utf-8 -*-
"""
Example datasets: cifar and mnist
"""
import gin
import logging
import os
import numpy as np
import h5py
from keras import datasets
from keras.datasets import mnist, fashion_mnist
from keras.utils import np_utils

from keras.preprocessing import sequence
from keras.datasets import imdb as load_imdb

from src import DATA_DIR

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
FMNIST_DIR = os.path.join(ROOT_DIR, "data/fmnist")

@gin.configurable
def cifar(which=10, preprocessing="center", seed=777, use_valid=True):
    rng = np.random.RandomState(seed)
    meta_data = {}

    if which == 10:
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    elif which == 100:
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    else:
        raise NotImplementedError(which)

    # Minor conversions
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    y_train = y_train.astype("long").reshape(-1,)
    y_test = y_test.astype("long").reshape(-1,)

    # Always outputs channels first
    if x_train.shape[-1] == 3:
        logging.info("Transposing")
        x_train = x_train.transpose((0, 3, 1, 2))
        x_test = x_test.transpose((0, 3, 1, 2))

    if use_valid:
        # Some randomization to make sure
        ids = rng.choice(len(x_train), len(x_train), replace=False)
        assert len(set(ids)) == len(ids) == len(x_train)
        x_train = x_train[ids]
        y_train = y_train[ids]

        N_valid = int(len(x_train) * 0.1)

        assert len(x_train) == 50000, len(x_train)
        assert N_valid == 5000

        (x_train, y_train), (x_valid, y_valid) = (x_train[0:-N_valid], y_train[0:-N_valid]), \
                                                 (x_train[-N_valid:], y_train[-N_valid:])

    meta_preprocessing = {"type": preprocessing}
    if preprocessing == "center":
        # This (I think) follows the original resnet paper. Per-pixel mean
        # and the global std, computed using the train set
        mean = np.mean(x_train, axis=0, keepdims=True)  # Pixel mean
        std = np.std(x_train)
        meta_preprocessing['mean'] = mean
        meta_preprocessing['std'] = std
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        if use_valid:
            x_valid = (x_valid - mean) / std
    elif preprocessing == "01":  # Required by scatnet
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        if use_valid:
            x_valid = x_valid / 255.0
    else:
        raise NotImplementedError("Not implemented preprocessing " + preprocessing)

    logging.info('x_train shape:' + str(x_train.shape))
    logging.info(str(x_train.shape[0]) + 'train samples')
    logging.info(str(x_test.shape[0]) + 'test samples')
    if use_valid:
        logging.info(str(x_valid.shape[0]) + 'valid samples')
    logging.info('y_train shape:' + str(y_train.shape))

    # Prepare test
    train = [x_train, y_train]
    test = [x_test, y_test]
    if use_valid:
        valid = [x_valid, y_valid]

    w, h, c = train[0].shape[1:4]
    meta_data['input_dim'] = (w, h, c)
    meta_data['preprocessing'] = meta_preprocessing

    if use_valid:
        return train, valid, test, meta_data
    else:
        return train, test, test, meta_data


@gin.configurable
def mnist(which="fmnist", preprocessing="01", seed=777, use_valid=True):
    """
    Returns
    -------
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
    """
    rng = np.random.RandomState(seed)
    meta_data = {}

    if use_valid:
        logger.info("Using valid")
    else:
        logger.info("Using as valid test")

    if which == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
    elif which == "fmnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train, y_train = np.array(X_train).astype("float32"), np.array(y_train)
        X_test, y_test = np.array(X_test).astype("float32"), np.array(y_test)
        x_train = X_train.reshape(-1, 1, 28, 28)
        x_test = X_test.reshape(-1, 1, 28, 28)
    else:
        raise NotImplementedError()

    y_train = y_train.astype("long").reshape(-1,)
    y_test = y_test.astype("long").reshape(-1,)

    # Permute
    ids_train = rng.choice(len(x_train), len(x_train), replace=False)
    ids_test = rng.choice(len(x_test), len(x_test), replace=False)
    x_train, y_train = x_train[ids_train], y_train[ids_train]
    x_test, y_test = x_test[ids_test], y_test[ids_test]

    logger.info("Loaded dataset using eval")

    if use_valid:
        assert len(x_train) == 60000, len(x_train)
        (x_train, y_train), (x_valid, y_valid) = (x_train[0:50000], y_train[0:50000]), \
                                                 (x_train[-10000:], y_train[-10000:])

    if preprocessing == "center":
        mean = np.mean(x_train, axis=0, keepdims=True)  # Pixel mean
        # Complete std as in https://github.com/gcr/torch-residual-networks/blob/master/data/cifar-dataset.lua#L3
        std = np.std(x_train)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        if use_valid:
            x_valid = (x_valid - mean) / std
    elif preprocessing == "01":  # Required by scatnet
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        if use_valid:
            x_valid = x_valid / 255.0
    else:
        raise NotImplementedError("Not implemented preprocessing " + preprocessing)

    logger.info('x_train shape:' + str(x_train.shape))
    logger.info(str(x_train.shape[0]) + 'train samples')
    logger.info(str(x_test.shape[0]) + 'test samples')
    if use_valid:
        logger.info(str(x_valid.shape[0]) + 'valid samples')
    logger.info('y_train shape:' + str(y_train.shape))

    # Prepare test
    train = [x_train, y_train]
    test = [x_test, y_test]
    if use_valid:
        valid = [x_valid, y_valid]

    w, h = meta_data['x_train'].shape[1:3]
    n_channels = meta_data['x_train'].shape[3]
    logger.info((w, h, n_channels))

    if use_valid:
        return train, valid, test, meta_data
    else:
        # Using as valid test
        return train, test, test, meta_data
