#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simple model definitions
"""

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def build_simple_model(config):
    model = Sequential()

    model.add(Conv2D(config['n_filters'], (3, 3), padding='same', data_format='channels_first',
        input_shape=(3, 32, 32)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(config['dim_dense']))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model