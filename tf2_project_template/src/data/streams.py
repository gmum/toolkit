# -*- coding: utf-8 -*-
"""
Streams used in the project (e.g. augmentation)
"""
import numpy as np

class DatasetGenerator(object):
    def __init__(self, dataset, seed, batch_size, shuffle=True):
        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        if self.shuffle:
            ids = self.rng.choice(len(self.dataset[0]), len(self.dataset[0]), replace=False)
        else:
            ids = np.arange(len(self.dataset[0]))
        self.dataset = [self.dataset[0][ids], self.dataset[1][ids]]
        def _iter():
            for id in range((len(self.dataset[0]) + self.batch_size - 1) // self.batch_size):
                yield self.dataset[0][id * self.batch_size:(id + 1) * self.batch_size], \
                      self.dataset[1][id * self.batch_size:(id + 1) * self.batch_size]
        return _iter()

    def __len__(self):
        return len(self.dataset[0])