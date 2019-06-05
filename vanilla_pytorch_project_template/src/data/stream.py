# -*- coding: utf-8 -*-
"""
Streams used in the project (e.g. augmentation)
"""
import os
import logging
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from src import DATA_FORMAT

logger = logging.getLogger(__name__)

def _to_generator(X, y, meta_data, datagen=None, batch_size=128, seed=777):
    """
    Returns a generator over the dataset
    """
    rng = np.random.RandomState(seed)

    X = X.copy()
    y = y.copy()

    if len(y.shape) == 2:
        assert y.shape[1] == meta_data['n_classes']

    if datagen is None:
        def generator():
            while True:
                ids = rng.choice(len(X), len(X), replace=False)
                assert len(set(ids)) == len(ids) == len(X)

                # Discards last batch of non-equal size
                batch = [[], []]

                for id in ids:
                    batch[0].append(X[id])
                    batch[1].append(y[id])

                    if len(batch[0]) == batch_size:
                        batch = [np.array(bb) for bb in batch]

                        yield batch

                        batch = [[], []]

        return generator()
    else:
        datagen.fit(X)
        return datagen.flow(X, y, batch_size=batch_size, shuffle=True, seed=seed)


def to_stream(ds, meta_data, augmentation, data_seed, batch_size):
    # Convert dataset to stream according to the configuration
    if augmentation == 'cifar':
        logger.info("Applying CIFAR augmentation")
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            data_format=DATA_FORMAT,
            width_shift_range=0.125,  # 4 px
            height_shift_range=0.125,  # 4 px
            horizontal_flip=True,
            vertical_flip=False)
    elif augmentation == "none":
        logger.info("No augmentation")
        datagen = None
    else:
        raise NotImplementedError("Not implemented augmentation " + str(augmentation))

    return _to_generator(ds[0], ds[1], datagen=datagen,
        meta_data=meta_data, batch_size=batch_size, seed=data_seed)
