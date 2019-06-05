# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid
"""
import os
from functools import partial

from src.data.datasets import cifar10, cifar100, fashion_mnist, mnist, imdb
from src.experimental import *
from .bias import bias
from .stream import to_stream

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
FMNIST_DIR = os.path.join(ROOT_DIR, "data/fmnist")


def random_label_corrupt(X, y, n_classes, random_Y_fraction=0.0, seed=777):
    """
    Params
    ------
    arrays: list of tuples
        List of (X, y) pairs to which add random labels
    """
    rng = np.random.RandomState(seed)
    n_random = int(random_Y_fraction * len(X))
    ids_random = rng.choice(len(X), n_random, replace=False)
    logger.info(y.shape)
    y[ids_random] = rng.randint(0, n_classes, size=(y[ids_random].shape[0],)).reshape(-1, 1)
    return X, y, ids_random



def init_data(config):
    ## Init transform ##
    if config.get('bias', '') != '':
        transform = partial(bias, bias=config['bias'], bias_kwargs=config['bias_kwargs'])
    else:
        transform = None

    ## Init dataset ##
    dataset, dataset_kwargs = config['dataset'], config['dataset_kwargs']
    init_kwargs = dict(dataset_kwargs)
    init_kwargs['which'] = dataset_kwargs.get('which', '')
    init_kwargs['cls'] = dataset_kwargs.get('cls', [])
    init_kwargs['seed'] = config['data_seed']
    if 'augmentation' in init_kwargs:
        del init_kwargs['augmentation']
    if 'n_examples' in init_kwargs:
        del init_kwargs['n_examples']
    if 'random_Y_fraction' in init_kwargs:
        del init_kwargs['random_Y_fraction']
    train, valid, test, meta_data = globals()[dataset](transform=transform, **init_kwargs)

    n_classes = meta_data['n_classes']

    ## Post-processing (after normalization) ##
    logging.info(n_classes)
    logging.info(valid[1].shape)
    logging.info(test[1].shape)

    # NOTE: I keep these transformations dynamical, because
    # it can get very messy storing datasets when there are
    # multiple seeds. I also do not cache, cause caching can get
    # even worse.

    if config.get('n_examples', -1) > 0:
        assert len(train[0]) >= config['n_examples']
        train = [train[0][0:config['n_examples']], train[1][0:config['n_examples']]]

    if dataset_kwargs.get('random_Y_fraction', 0) != 0.:
        logger.info("Adding random labels {}".format(config['random_Y_fraction']))
        X_train, y_train, ids_random_train = random_label_corrupt(train[0], train[1], n_classes,
            random_Y_fraction=config['random_Y_fraction'],
            seed=config['data_seed'])
        train = [X_train, y_train]
        meta_data['ids_random_train'] = ids_random_train

    # Configure stream + optionally augmentation
    meta_data['x_train'] = train[0]
    meta_data['y_train'] = train[1]

    # Assuming static batch size
    meta_data['input_dim_with_bs'] = [config['batch_size']] + list(valid[0].shape[1:])
    meta_data['output_dim_with_bs'] = [config['batch_size']] + list(valid[1].shape[1:])

    if len(valid[1].shape) == 2:
        assert valid[1].shape[1] == n_classes
        assert test[1].shape[1] == n_classes
        assert train[1].shape[1] == n_classes

    train_stream = to_stream(config, train, meta_data)
    train_stream_duplicated = to_stream(config, train, meta_data)
    # Add to meta_data a single pass through an augmentated dataset
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
    return train_stream, valid, test, meta_data
