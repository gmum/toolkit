# -*- coding: utf-8 -*-
"""
Callback implementation (inspired by Keras).
"""

# NOTE(kudkudak): There is no (yet) standalone tensorboard, and I don't think it makes sense to use tensorboardX
import tensorflow

from src.utils import save_weights

import sys
import numpy as np
import os
import pickle
import logging
import time
import datetime
import json

logger = logging.getLogger(__name__)

# TODO: Add common args here
class Callback(object):
    """
    Attributes:
        params (dict): Contains a key 'epoch' and a key 'steps_per_epoch'
            which are passed to the `fit` function in `Model`. It may
            contain other keys.
        model (Model): a reference to the `Model` object which is using the
            callback.
    """

    def __init__(self):
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs):
        """
        Is called before the begining of each epoch.

        Args:
            epoch (int): The epoch number.
            logs (dict): Usually an empty dict.
        """
        pass

    def on_epoch_end(self, epoch, logs):
        """
        Is called before the end of each epoch.

        Args:
            epoch (int): The epoch number.
            logs (dict): Contains the following keys:

                 * 'epoch': The epoch number.
                 * 'loss': The average loss of the batches.
                 * Other metrics: One key for each type of metrics. The metrics
                   are also averaged.
                 * val_loss': The average loss of the batches on the validation
                   set.
                 * Other metrics: One key for each type of metrics on the
                   validation set. The metrics are also averaged.

        Example::

            logs = {'epoch': 6, 'loss': 4.34462, 'accuracy': 0.766, 'val_loss': 5.2352, 'val_accuracy': 0.682}
        """
        pass

    def on_batch_begin(self, batch, logs):
        """
        Is called before the begining of each batch.

        Args:
            batch (int): The batch number.
            logs (dict): Usually an empty dict.
        """
        pass

    def on_batch_end(self, batch, logs):
        """
        Is called before the end of each batch.

        Args:
            batch (int): The batch number.
            logs (dict): Contains the following keys:

                 * 'batch': The batch number.
                 * 'loss': The loss of the batch.
                 * Other metrics: One key for each type of metrics.

        Example::

            logs = {'batch': 6, 'loss': 4.34462, 'accuracy': 0.766}
        """
        pass

    def on_backward_end(self, batch):
        """
        Is called after the backpropagation but before the optimization step.

        Args:
            batch (int): The batch number.
        """
        pass

    def on_train_begin(self, logs):
        """
        Is called before the begining of the training.

        Args:
            logs (dict): Usually an empty dict.
        """
        pass

    def on_train_end(self, logs):
        """
        Is called before the end of the training.

        Args:
            logs (dict): Usually an empty dict.
        """
        pass



class LRSchedule(Callback):
    def __init__(self, lr_schedule):
        self.lr_schedule = lr_schedule

    def on_epoch_begin(self, epoch, logs):
        for e, v in self.lr_schedule:
            if epoch < e:
                break
        for group in self.model.optimizer.param_groups:
            group['lr'] = v
        logger.info("Fix learning rate to {}".format(v))


class History(Callback):
    """Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def __init__(self, save_path=None):
        self.save_path = save_path
        super(History, self).__init__()

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        if self.save_path is not None:
            logger.info("Saving history to " + self.save_path)
            pickle.dump(self.epoch, open(self.save_path, "wb"))

class ModelCheckpoint(Callback):
    def __init__(self, filepath, model, optimizer, monitor='val_loss', verbose=0,
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


class DumpTensorboardSummaries(Callback):
    def __init__(self, save_path):
        self._save_path = save_path
        super(DumpTensorboardSummaries, self).__init__()

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


class MetaSaver(Callback):
    def __init__(self, save_path, config):
        self.save_path = save_path
        self.config = config

    def on_train_begin(self, logs=None):
        if os.path.exists(os.path.join(self.save_path, "FINISHED")):
            logger.info("Finished training. Exiting. Remove FINISHED file if you want to train anyways.")
            exit(0)

        assert os.system("cp {} {}".format(sys.argv[0], self.save_path)) == 0, "Failed to execute cp of source script"

        utc_date = datetime.datetime.utcnow().strftime("%Y_%m_%d")

        time_start = time.time()
        cmd = "python " + " ".join(sys.argv)
        self.meta = {"cmd": cmd,
                     "save_path": self.save_path,
                     "start_utc_date": utc_date,
                     "execution_time": -time_start}

        json.dump(self.config, open(os.path.join(self.save_path, "config.json"), "w"), indent=4)
        json.dump(self.meta, open(os.path.join(self.save_path, "meta.json"), "w"), indent=4)

    def on_train_end(self, logs=None):
        self.meta['execution_time'] += time.time()
        json.dump(self.meta, open(os.path.join(self.save_path, "meta.json"), "w"), indent=4)
        os.system("touch " + os.path.join(self.save_path, "FINISHED"))
