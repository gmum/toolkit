# -*- coding: utf-8 -*-
"""
Utils for datasets.
"""
import gin
import logging
import numpy as np

from src import DATA_FORMAT
# from src.data.streams import DatasetGenerator
from src import DATA_DIR, DATA_FORMAT, DATA_NUM_WORKERS

from os.path import join

logger = logging.getLogger(__name__)

import torchvision.transforms as T
import torch
from torch.utils.data import Sampler
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

import numpy as np


class DatasetFromNumpy(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, index):
        x, y = self.ds[0][index], self.ds[1][index]

        return x, y

    def __len__(self):
        assert len(self.ds[0]) == len(self.ds[1])
        return len(self.ds[0])


class TransformedDataset(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.ds[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.ds)


class ShuffledDataset(TransformedDataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, ds, rand_frac, n_classes, transform=None, seed=0):
        super().__init__(ds, transform=transform)
        self.create_shuffled(rand_frac, n_classes, seed)

    def create_shuffled(self, rand_frac, n_classes, seed):
        # TOD: use the same random state
        rng = np.random.RandomState(seed)

        inds = self.ds.indices
        targets = np.array(self.ds.dataset.targets)
        n = len(inds)
        n_rand = int(n * rand_frac)

        # TODO: make it work for arbitrarily many classes
        rand_inds = rng.choice(inds, n_rand, replace=False)
        rand_labels = rng.randint(0, n_classes, (n_rand,))
        targets[rand_inds] = rand_labels
        self.ds.dataset.targets = targets.tolist()

        real_inds = np.array(list(set(inds) - set(rand_inds)))

        self.real_ds = torch.utils.data.Subset(self.ds.dataset, real_inds)
        self.real_data = TransformedDataset(self.real_ds, self.transform)
        self.noisy_ds = torch.utils.data.Subset(self.ds.dataset, rand_inds)
        self.noisy_data = TransformedDataset(self.noisy_ds, self.transform)


def construct_generators_and_meta(train, valid, test, seed, batch_size, stream_seed, workers=DATA_NUM_WORKERS,
        rand_frac=0, pin_memory=True):
    """
    A helper function that converts data sources into generators and meta data usable with the rest of the code

    Returns: a tuple of (train_generator, valid_generator, test_generator, meta_data)

    Note
    ----
    Assumes labels are one hot encoded.
    """
    rng_stream = np.random.RandomState(stream_seed)

    train_generator = construct_generator(train, workers=workers, shuffle=True, batch_size=batch_size, rng=rng_stream,
        pin_memory=pin_memory)
    train_duplicated_generator = construct_generator(train, workers=1, shuffle=True, batch_size=batch_size,
        rng=rng_stream, pin_memory=pin_memory)
    valid_generator = construct_generator(valid, workers=workers, shuffle=False, batch_size=batch_size,
        pin_memory=pin_memory)
    valid_duplicated_generator = construct_generator(valid, workers=workers, shuffle=False, batch_size=batch_size,
        pin_memory=pin_memory)
    test_generator = construct_generator(test, 1, shuffle=False, batch_size=batch_size, pin_memory=pin_memory)

    meta_data = construct_meta_data(train=train, valid=valid, test=test,
        train_duplicated_generator=train_duplicated_generator,
        valid_duplicated_generator=valid_duplicated_generator)

    return train_generator, valid_generator, test_generator, meta_data


def construct_meta_data(train, valid, test,
        train_duplicated_generator,
        valid_duplicated_generator):
    meta_data = {}
    meta_data['train_ds'] = train
    meta_data['valid_ds'] = valid
    meta_data['test_ds'] = test
    meta_data['n_examples_train'] = len(train)
    meta_data['n_examples_valid'] = len(valid)
    meta_data['n_examples_test'] = len(test)
    meta_data['train_stream_duplicated'] = train_duplicated_generator
    meta_data['valid_stream_duplicated'] = valid_duplicated_generator
    meta_data['input_dim'] = meta_data['input_shape'] = train[0][0].shape  # Take first element and the image of it
    if isinstance(train[0][1], int) or isinstance(train[0][1], float):
        max_label = 0
        # TODO: This might be slow for large datasets
        for x, y in train:
            max_label = max(max_label, y)
        meta_data['num_classes'] = max_label
        meta_data['one_hot'] = False
    else:
        meta_data['num_classes'] = len(train[0][1])  # We assume labels are one hot encoded
        meta_data['one_hot'] = True
    return meta_data


def construct_generator(ds, workers, shuffle, batch_size, rng=None, pin_memory=True):
    # TODO: Is this collate function the optimal way to do it?
    # TODO: Generalize this. It works only for specific shapes of x and y
    def collate_fn(xy):
        if hasattr(xy[0][0], 'numpy'):
            return (torch.stack([x for x, y in xy]), torch.tensor([y for x, y in xy]))
        else:
            return (np.stack([x for x, y in xy]), np.array([y for x, y in xy]))


    class SeededRandomSampler(Sampler):
        r"""Default PyTorch sampler is not seeded.
        """

        def __init__(self, data_source, rng, num_samples=None):
            self.data_source = data_source
            self.rng = rng
            self._num_samples = num_samples

        @property
        def num_samples(self):
            if self._num_samples is None:
                return len(self.data_source)
            return self._num_samples

        def __iter__(self):
            n = len(self.data_source)
            return iter(self.rng.choice(n, n, replace=False))

        def __len__(self):
            return self.num_samples

    if shuffle:
        sampler = SeededRandomSampler(data_source=ds, rng=rng)
    else:
        sampler = None
    # pin_memory should just speedup
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=workers, sampler=sampler,
        pin_memory=pin_memory)
