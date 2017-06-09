#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains simple CNN on cifar10/cifar100
"""
import cPickle
import json
import logging
import os
import sys
import argh  # pip install argh --user
import keras.backend as K
import numpy as np
import theano
import theano.tensor as T

from keras.backend.common import _EPSILON
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, LambdaCallback
from keras.datasets import cifar10, cifar100
from keras.engine import merge, Input, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, AveragePooling2D, BatchNormalization, \
    Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2 as l2_reg
from keras.utils import np_utils

from src.layers import SkipForward, residual_block
from src.keras_utils import *
from src.utils import utc_timestamp, kwargs_namer, configure_logger, cos_loss

from vegab import main, MetaSaver, AutomaticNamer

def train(config, save_path):
    # Load data

    # Load model

    # Optimizer

    # Call training loop

if __name__ == "__main__":
    main(train, plugins=[MetaSaver(), AutomaticNamer(prefix="timestamp_namer")])