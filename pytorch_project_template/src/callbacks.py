# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid
"""

from pytoune.framework.callbacks import Callback

# For Tensorboard only
# Might misbehave with tensorflow-gpu, make sure u use tensorflow-cpu
try:
    import tensorflow
except:
    pass

from src.utils import save_weights

import pandas as pd
import numpy as np
import os
import pickle
import logging
logger = logging.getLogger(__name__)

class ModelCheckpoint(Callback):
    def __init__(self, filepath, model, optimizer,monitor='val_loss', verbose=0,
            save_best_only=False,
            mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.optimizer = optimizer
        self.verbose = verbose
        self.filepath = filepath
        self.model = model
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        del state['optimizer']
        return state

    def __setstate__(self, newstate):
        newstate['model'] = self.model
        newstate['optimizer'] = self.optimizer
        self.__dict__.update(newstate)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                            current, self.filepath))
                        self.best = current
                        save_weights(self.model.model, self.optimizer, self.filepath)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, self.filepath))
                    save_weights(self.model.model, self.optimizer, self.filepath)

class LambdaCallback(Callback):
    def __init__(self,
                 on_epoch_begin=None,
                 on_epoch_end=None,
                 on_batch_begin=None,
                 on_batch_end=None,
                 on_train_begin=None,
                 on_train_end=None,
                 **kwargs):
        super(LambdaCallback, self).__init__()
        self.__dict__.update(kwargs)
        if on_epoch_begin is not None:
            self.on_epoch_begin = on_epoch_begin
        else:
            self.on_epoch_begin = lambda epoch, logs: None
        if on_epoch_end is not None:
            self.on_epoch_end = on_epoch_end
        else:
            self.on_epoch_end = lambda epoch, logs: None
        if on_batch_begin is not None:
            self.on_batch_begin = on_batch_begin
        else:
            self.on_batch_begin = lambda batch, logs: None
        if on_batch_end is not None:
            self.on_batch_end = on_batch_end
        else:
            self.on_batch_end = lambda batch, logs: None
        if on_train_begin is not None:
            self.on_train_begin = on_train_begin
        else:
            self.on_train_begin = lambda logs: None
        if on_train_end is not None:
            self.on_train_end = on_train_end
        else:
            self.on_train_end = lambda logs: None

class DumpTensorflowSummaries(Callback):
    def __init__(self, save_path):
        self._save_path = save_path
        super(DumpTensorflowSummaries, self).__init__()

    @property
    def file_writer(self):
        if not hasattr(self, '_file_writer'):
            self._file_writer = tensorflow.summary.FileWriter(
                self._save_path, flush_secs=10.)
        return self._file_writer

    def on_epoch_end(self, epoch, logs=None):
        summary = tensorflow.Summary()
        for key, value in logs.items():
            try:
                float_value = float(value)
                value = summary.value.add()
                value.tag = key
                value.simple_value = float_value
            except:
                pass
        self.file_writer.add_summary(
            summary, epoch)