# -*- coding: utf-8 -*-
"""
Datasets used in the project.

Each dataset is returned as a tuple (train_loader, valid_loader, test_loader, meta_data)

Uses PyTorch due to the widespread code for dataset handling in PyTorch and standardized augmentations
for image datasets in particular.
"""
import logging
import gin
import numpy as np
import torch
import torchvision.transforms as T
import torchvision
from PIL import Image
import PIL

from os.path import join
from torch.utils.data import Subset

from src import DATA_DIR, DATA_NUM_WORKERS
from src.data.utils import construct_generators_and_meta, TransformedDataset, ShuffledDataset, DatasetFromNumpy


logger = logging.getLogger(__name__)

TINY_IMAGENET_PATH = join(DATA_DIR, 'tiny-imagenet-200')

@gin.configurable
def cifar(use_valid, seed=777, stream_seed=777, variant="10", augment=True, batch_size=128, rand_frac=0):
    rng = np.random.RandomState(seed)
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    if augment is True:
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(cifar_mean, cifar_std),
        ])
    else:
        transform_train = T.Compose([
            T.ToTensor(),
            T.Normalize(cifar_mean, cifar_std),
        ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(cifar_mean, cifar_std),
    ])

    if variant == '10':
        n_classes = 10
        def to_one_hot(y):
            # CHANGE N CLASSES
            one_hot = np.zeros(shape=(10,))
            one_hot[y] = 1
            one_hot = torch.tensor(one_hot)
            return one_hot
        trainval = torchvision.datasets.CIFAR10(
            root=join(DATA_DIR, 'cifar100'), train=True, download=True, transform=None, target_transform=None)
        test = torchvision.datasets.CIFAR10(
            root=join(DATA_DIR, 'cifar100'), train=False, download=True, transform=None, target_transform=None)
        assert len(trainval) == 50000
    elif variant in {'100', '100a', '100b'}:
        n_classes = 100
        def to_one_hot(y):
            # CHANGE N CLASSES
            one_hot = np.zeros(shape=(100,))
            one_hot[y] = 1
            one_hot = torch.tensor(one_hot)
            return one_hot
        trainval = torchvision.datasets.CIFAR100(
            root=join(DATA_DIR, 'cifar100'), train=True, download=True, transform=None, target_transform=None)
        test = torchvision.datasets.CIFAR100(
            root=join(DATA_DIR, 'cifar100'), train=False, download=True, transform=None, target_transform=None)
        assert len(trainval) == 50000
    elif variant in {'100c', '100d'}:
        n_classes = 100
        def to_one_hot(y):
            # CHANGE N CLASSES
            one_hot = np.zeros(shape=(100,))
            one_hot[y] = 1
            one_hot = torch.tensor(one_hot)
            return one_hot
        trainval = torchvision.datasets.CIFAR100(
            root=join(DATA_DIR, 'cifar100'), train=True, download=True, transform=None, target_transform=None)
        test = torchvision.datasets.CIFAR100(
            root=join(DATA_DIR, 'cifar100'), train=False, download=True, transform=None, target_transform=None)
        assert len(trainval) == 50000
    else:
        raise NotImplementedError()

    # Split here to ensure different validation set
    ids = np.arange(len(trainval))
    rng.shuffle(ids)
    if variant.endswith("b"):
        trainval = Subset(trainval, ids[len(trainval)//2:])
    elif variant.endswith("a"):
        trainval = Subset(trainval, ids[0:len(trainval)//2])

    if use_valid:
        ids = rng.choice(len(trainval), len(trainval), replace=False)
        N_valid = int(len(trainval) * 0.1)
        ids_train, ids_val = ids[0:-N_valid], ids[-N_valid:]
        train, valid = Subset(trainval, ids_train), Subset(trainval, ids_val)
        assert len(valid) == int(0.1 * len(trainval)) and len(train) == int(0.9 * len(trainval))
    else:
        train, valid = trainval, test

    # Same valid, weirdly but OK.
    ids = np.arange(len(train))
    rng.shuffle(ids)
    if variant.endswith("c"):
        train = Subset(train, ids[len(train)//2:])
    elif variant.endswith("d"):
        train = Subset(train, ids[0:len(train)//2])

    if rand_frac > 0:
        train = ShuffledDataset(train, rand_frac=rand_frac, n_classes=n_classes, transform=transform_train, seed=seed)
    else:
        train = TransformedDataset(train, transform=transform_train)
    test = TransformedDataset(test, transform=transform_test)
    valid = TransformedDataset(valid, transform=transform_test)

    return construct_generators_and_meta(train, valid, test, batch_size=batch_size, seed=seed, stream_seed=stream_seed,
                                         workers=DATA_NUM_WORKERS)

@gin.configurable
def stl10(use_valid, seed=777, stream_seed=777, augment=True, batch_size=128):
    rng = np.random.RandomState(seed)
    resize = T.Lambda(lambda x: x.resize((32, 32), resample=PIL.Image.BOX))

    def to_one_hot(y):
        one_hot = np.zeros(shape=(10,))
        one_hot[y] = 1
        one_hot = torch.tensor(one_hot)
        return one_hot
    if augment is True:
        transform_train = [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ]
    else:
        transform_train = [
            T.ToTensor()
        ]
    transform_test = [
        T.ToTensor(),
    ]

    trainval = torchvision.datasets.STL10(
        root=join(DATA_DIR, 'stl10'), split="train", download=True, transform=resize, target_transform=None)
    test = torchvision.datasets.STL10(
        root=join(DATA_DIR, 'stl10'), split="test", download=True, transform=resize, target_transform=None)

    assert len(trainval) == 5000

    if use_valid:
        ids = rng.choice(len(trainval), len(trainval), replace=False)
        N_valid = int(len(trainval) * 0.1)
        ids_train, ids_val = ids[0:-N_valid], ids[-N_valid:]
        train, valid = Subset(trainval, ids_train), Subset(trainval, ids_val)
    else:
        train, valid = trainval, test

    # Compute the standard channel-wise normalization by quickly loading dataset to memory
    X = [T.ToTensor()(train[i][0]).numpy() for i in range(len(train))]
    X = np.array(X)
    assert X.shape[1] == 3, X.shape
    stl10_mean = np.mean(X, axis=(0, 2, 3))
    stl10_std = np.std(X, axis=(0, 2, 3))
    transform_test.append(T.Normalize(stl10_mean, stl10_std))
    transform_train.append(T.Normalize(stl10_mean, stl10_std))

    train = TransformedDataset(train, transform=T.Compose(transform_train))
    test = TransformedDataset(test, transform=T.Compose(transform_test))
    valid = TransformedDataset(valid, transform=T.Compose(transform_test))

    return construct_generators_and_meta(train, valid, test, batch_size=batch_size, seed=seed, stream_seed=stream_seed,
                                         workers=DATA_NUM_WORKERS)



if __name__ == "__main__":
    train, valid, test, meta_data = cifar(seed=1, stream_seed=1, batch_size=128, augment=False, use_valid=True)
    for x, y in train:
        break

    x = x.numpy()

    import matplotlib.pylab as plt
    for xx in x[0:4]:
        plt.imshow(xx.transpose(1, 2, 0))
        plt.show()
        plt.close()
    xx = []
    for x, y in train:
        xx.append(x)
        if len(xx) > 40:
            break
    print("BATCH SHAPE")
    print(x.shape)
    print("MEAN")
    assert np.concatenate(xx, axis=0).shape[1] == 3
    print(np.mean(np.concatenate(xx, axis=0), axis=(0, 2, 3)))
    print("STD")
    print(np.std(np.concatenate(xx, axis=0), axis=(0, 2, 3)))
    print("MAX")
    print(np.max(np.max(np.concatenate(xx, axis=0), axis=0)))
    print("MIN")
    print(np.min(np.min(np.concatenate(xx, axis=0), axis=0)))
