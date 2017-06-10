# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid
"""
from keras.datasets import cifar10, cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

import logging
logging.getLogger(__name__)

def get_cifar(dataset="cifar10", data_format="channels_first", augmented=False, batch_size=128, preprocessing="center"):
    """
    Returns train iterator and test X, y.
    """

    # the data, shuffled and split between train and test sets
    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        raise NotImplementedError()

    if x_train.shape[3] == 3:
        logging.info("Transposing")
        x_train = x_train.transpose((0, 3, 1, 2))[0:100] # For speed
        x_test = x_test.transpose((0, 3, 1, 2))[0:100] # For speed
    assert x_train.shape[1] == 3

    if preprocessing == "center":
        mean = np.mean(x_train, axis=0, keepdims=True)
        std = np.std(x_train)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
    elif preprocessing == "01": # Required by scatnet
        x_train = x_train / 255.0
        x_test = x_test / 255.0
    else:
        raise NotImplementedError("Not implemented preprocessing " + preprocessing)

    logging.info('x_train shape:' + str(x_train.shape))
    logging.info(str(x_train.shape[0]) + 'train samples')
    logging.info(str(x_test.shape[0]) + 'test samples')

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train)[0:100]
    y_test = np_utils.to_categorical(y_test)[0:100]

    train, test = None, [x_test, y_test]
    if augmented:
        datagen_train = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            data_format=data_format,
            width_shift_range=0.125,
            height_shift_range=0.125,
            horizontal_flip=True,
            vertical_flip=False)
        datagen_train.fit(x_train)
        train = datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
    else:
        raise NotImplementedError()

    return train, test, {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
