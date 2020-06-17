# -*- coding: utf-8 -*-
"""
Datasets used in the project
"""
import gin
import logging
import numpy as np
from tensorflow.keras import datasets
from keras.utils import to_categorical

from src import DATA_FORMAT
from src.data.streams import DatasetGenerator

logger = logging.getLogger(__name__)

def _to_streams(train, valid, test, meta_data, n_examples, seed, batch_size, num_classes=None):
    if n_examples > 0:
        assert len(train[0]) >= n_examples
        train = [train[0][0:n_examples], train[1][0:n_examples]]

    meta_data['x_train_raw'] = train[0]
    meta_data['y_train_raw'] = train[1]
    meta_data['x_valid'] = valid[0]
    meta_data['y_valid'] = valid[1]
    meta_data['input_dim'] = list(train[0].shape[1:])
    meta_data['input_shape'] = list(train[0].shape[1:])
    meta_data['n_examples_train'] = len(train[0])
    meta_data['n_examples_valid'] = len(valid[0])
    meta_data['n_examples_test'] = len(test[0])

    if num_classes is None:
        if train[1].ndim == 1:
            meta_data['num_classes'] = train[1].max() + 1
        else:
            meta_data['num_classes'] = train[1].shape[1]
    else:
        meta_data['num_classes'] = num_classes

    logger.info(f"Creating stream from dataset of shape {train[0].shape},{train[1].shape}")

    train_stream = DatasetGenerator(train, seed=seed, batch_size=batch_size, shuffle=True)

    # Save some extra versions of the dataset. Just a pattern that is useful.
    train_stream_duplicated = DatasetGenerator(train, seed=seed, batch_size=batch_size, shuffle=True)
    x_train_aug, y_train_aug = [], []
    n = 0

    for x, y in train_stream_duplicated:
        if not isinstance(x, np.ndarray):
            x = x.cpu().numpy()
            y = x.cpu().numpy()

        x_train_aug.append(x)
        y_train_aug.append(y)
        n += len(x)
        if n >= len(meta_data['x_train_raw']):
            break
    meta_data['x_train_aug'] = np.concatenate(x_train_aug, axis=0)[0:len(meta_data['x_train_raw'])]
    meta_data['y_train_aug'] = np.concatenate(y_train_aug, axis=0)[0:len(meta_data['x_train_raw'])]
    meta_data['train_stream_duplicated'] = train_stream_duplicated

    # Return
    valid = DatasetGenerator(valid, seed=seed, batch_size=batch_size, shuffle=False)
    test = DatasetGenerator(test, seed=seed, batch_size=batch_size, shuffle=False)
    return [train_stream, valid, test, meta_data]



@gin.configurable
def cifar(seed, batch_size, n_examples=-1, stream_seed=1, variant=10, preprocessing="center", use_valid=True,
          one_hot=False,
          augment=False, format="default"):
    logger.warning("Using keras based CIFAR. Slow for small models with augmentation!")
    # TODO: Consider using prefetching

    rng = np.random.RandomState(seed)
    # TODO: Add raise

    if variant == 10:
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    elif variant == "10a":
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        x_train = x_train[0:len(x_train) // 2]
        y_train = y_train[0:len(y_train) // 2]
    elif variant == "10b":
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        x_train = x_train[len(x_train) // 2:]
        y_train = y_train[len(y_train) // 2:]
    elif variant == 100:
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    elif variant == "100a":
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        x_train = x_train[0:len(x_train) // 2]
        y_train = y_train[0:len(y_train) // 2]
    elif variant == "100b":
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        x_train = x_train[len(x_train) // 2:]
        y_train = y_train[len(y_train) // 2:]
    else:
        raise NotImplementedError(variant)

    # Minor conversions
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    y_train = y_train.astype("long").reshape(-1, )
    y_test = y_test.astype("long").reshape(-1, )

    if one_hot:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    # Always outputs channels first
    if x_train.shape[-1] == 3 and (DATA_FORMAT == "channels_first" or format == "channels_first"):
        logging.info("Transposing")
        x_train = x_train.transpose((0, 3, 1, 2))
        x_test = x_test.transpose((0, 3, 1, 2))
    elif x_train.shape[1] == 3 and (DATA_FORMAT == "channels_last" or format == "channels_last"):
        logging.info("Transposing")
        x_train = x_train.transpose((0, 2, 3, 1))
        x_test = x_test.transpose((0, 2, 3, 1))

    if use_valid:
        # Some randomization to make sure
        ids = rng.choice(len(x_train), len(x_train), replace=False)
        assert len(set(ids)) == len(ids) == len(x_train)
        x_train = x_train[ids]
        y_train = y_train[ids]

        N_valid = int(len(x_train) * 0.1)

        assert len(x_train) == 50000 or len(x_train) == 25000, len(x_train)
        assert N_valid == 5000 or N_valid == 2500

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

    test = [x_test, y_test]
    train = [x_train, y_train]
    if use_valid:
        valid = [x_valid, y_valid]

    print("Augment or not?")
    if augment:
        if n_examples > 0:
            assert len(train[0]) >= n_examples
            train = [train[0][0:n_examples], train[1][0:n_examples]]

        meta_data = {"n_examples_train": len(train[0])}

        def _get_gen():
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                data_format='channels_first',
                fill_mode='constant',
                width_shift_range=0.125,  # 4 px
                height_shift_range=0.125,  # 4 px
                horizontal_flip=True,
                vertical_flip=False)
            datagen.fit(x_train)
            return datagen

        train_loader = _get_gen().flow(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
        train_duplicated_loader = _get_gen().flow(x_train, y_train, batch_size=batch_size, shuffle=True, seed=seed)
        if use_valid:
            meta_data['x_valid'] = valid[0]
            meta_data['y_valid'] = valid[1]
            valid = DatasetGenerator(valid, seed=seed, batch_size=batch_size, shuffle=False)
        meta_data['x_test'] = test[0]
        meta_data['y_test'] = test[1]
        test = DatasetGenerator(test, seed=seed, batch_size=batch_size, shuffle=False)

        # Prepare meta_data
        meta_data['x_train_raw'] = train[0]
        meta_data['y_train_raw'] = train[1]
        meta_data['input_dim'] = meta_data['input_shape'] = list(train[0].shape[1:])
        print(meta_data['input_shape'])
        if train[1].ndim == 1:
            meta_data['num_classes'] = train[1].max() + 1
        else:
            meta_data['num_classes'] = train[1].shape[1]
        x_train_aug, y_train_aug = [], []
        n = 0
        for x, y in train_duplicated_loader:
            x_train_aug.append(x)
            y_train_aug.append(y)
            n += len(x)
            if n >= len(meta_data['x_train_raw']):
                break
        meta_data['x_train_aug'] = np.concatenate(x_train_aug, axis=0)[0:len(meta_data['x_train_raw'])]
        meta_data['y_train_aug'] = np.concatenate(y_train_aug, axis=0)[0:len(meta_data['x_train_raw'])]
        print(f"X mean {np.mean(meta_data['x_train_aug'])}")
        meta_data['train_stream_duplicated'] = train_duplicated_loader

        if use_valid:
            return [train_loader, valid, test, meta_data]
        else:
            return [train_loader, test, test, meta_data]

    else:
        logger.warning("No augment!")
        meta_data = {}
        if use_valid:
            return _to_streams(train, valid, test, meta_data,
                               n_examples=n_examples, seed=stream_seed, batch_size=batch_size)
        else:
            return _to_streams(train, test, test, meta_data,
                               n_examples=n_examples, seed=stream_seed, batch_size=batch_size)



if __name__ == "__init__":
    from src import configure_logger
    configure_logger('')
    train, valid, test, meta_data = cifar(seed=1, batch_size=128)
