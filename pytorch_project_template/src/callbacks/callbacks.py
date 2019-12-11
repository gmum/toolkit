# -*- coding: utf-8 -*-
"""
Callbacks implementation. Inspired by Keras.
"""

# NOTE(kudkudak): There is no (yet) standalone tensorboard, and I don't think it makes sense to use tensorboardX
import tensorflow

from src.utils import save_weights
from collections import defaultdict

import gin
import sys
import numpy as np
import os
import pickle
import logging
import time
import datetime
import json

logger = logging.getLogger(__name__)


class Callback(object):
    def __init__(self):
        pass

    def set_config(self, config):
        self.config = config

    def set_meta_data(self, meta_data):
        self.meta_data = meta_data

    def set_save_path(self, save_path):
        self.save_path = save_path

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_model(self, model, ignore=True):
        if ignore:
            return
        self.model = model

    def set_params(self, params):
        self.params = params

    def get_config(self):
        return self.config

    def get_meta_data(self):
        return self.meta_data

    def get_optimizer(self):
        return self.optimizer

    def get_params(self):
        return self.params

    def get_model(self):
        return self.model

    def get_save_path(self):
        return self.save_path

    def on_epoch_begin(self, epoch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        pass

    def on_batch_begin(self, batch, logs):
        pass

    def on_batch_end(self, batch, logs):
        pass

    def on_train_begin(self, logs):
        pass

    def on_train_end(self, logs):
        pass


class BaseLogger(Callback):
    """Callback that accumulates epoch averages."""
    def __init__(self):
        super(BaseLogger, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = defaultdict(float)

    def on_batch_end(self, batch, logs=None):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        if logs is not None:
            for k, v in logs.items():
                self.totals[k] += v * batch_size


    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.totals:
                logs[k] = self.totals[k] / self.seen


@gin.configurable
class LRSchedule(Callback):
    def __init__(self, base_lr, schedule):
        self.schedule = schedule
        self.base_lr = base_lr
        super(LRSchedule, self).__init__()

    def on_epoch_begin(self, epoch, logs):
        # Epochs starts from 0
        for e, v in self.schedule:
            if epoch < e:
                break
        for group in self.optimizer.param_groups:
            group['lr'] = v * self.base_lr
        logger.info("Set learning rate to {}".format(v * self.base_lr))


class History(Callback):
    """
    History callback.

    By default saves history every epoch, can be configured to save also every k examples
    """

    def __init__(self, save_every_k_examples=-1):
        self.examples_seen = 0
        self.save_every_k_examples = save_every_k_examples
        self.examples_seen_since_last_population = 0
        super(History, self).__init__()

    def on_train_begin(self, logs=None):
        # self.epoch = []
        self.history = {}
        self.history_batch = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        if self.save_path is not None:
            pickle.dump(self.history, open(os.path.join(self.save_path, "history.pkl"), "wb"))
            if self.save_every_k_examples != -1:
                pickle.dump(self.history_batch, open(os.path.join(self.save_path, "history_batch.pkl"), "wb"))

    def on_batch_end(self, batch, logs=None):
        # Batches starts from 1
        if self.save_every_k_examples != -1:
            if getattr(self.model, "history_batch", None) is None:
                setattr(self.model, "history_batch", self)
            assert "size" in logs
            self.examples_seen += logs['size']
            logs['examples_seen'] = self.examples_seen
            self.examples_seen_since_last_population += logs['size']

            if self.examples_seen_since_last_population > self.save_every_k_examples:
                for k, v in logs.items():
                    self.history_batch.setdefault(k, []).append(v)
                self.examples_seen_since_last_population = 0


class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
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
                        save_weights(self.model, self.optimizer, self.filepath)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, self.filepath))
                    save_weights(self.model, self.optimizer, self.filepath)


class LambdaCallback(Callback):
    def __init__(self,
                 on_epoch_begin=None,
                 on_epoch_end=None,
                 on_batch_begin=None,
                 on_batch_end=None,
                 on_train_begin=None,
                 on_train_end=None):
        super(LambdaCallback, self).__init__()
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


class CallbackPickableEveryKExamples(Callback):
    """
    A callback to run a given lambda function with a specified frequency
    """

    def __init__(self,
                 frequency="128ex",
                 on_batch_begin=False,
                 epoch_size=None,
                 name=None,
                 **kwargs):
        super(CallbackPickableEveryKExamples, self).__init__()
        logger.info("Construct callback name={} with frequency={}".format(name, str(frequency)))
        assert name is not None

        self.__dict__.update(kwargs)
        self.examples_seen = 0
        self.call_on_batch_begin = on_batch_begin
        self.epoch_size = epoch_size
        self.name = name
        self.examples_seen_since_last_call = 0
        if frequency.endswith("ex"):
            self.threshold = int(frequency[0:-2])
        elif frequency.endswith("exp"):
            self.threshold = int(self.epoch_size * float(frequency[0:-3]))
        else:
            raise NotImplementedError()
        self._epoch_logs = []
        self.calls = 0

    def on_batch_k_examples(self, batch, logs):
        raise NotImplementedError()

    def _call(self, batch, logs=None):
        assert "size" in logs
        self.examples_seen += logs['size']
        self.examples_seen_since_last_call += logs['size']

        # Always call on batch 0
        if (self.calls == 0) or (self.examples_seen_since_last_call >= self.threshold):
            t_start = time.time()

            logs_callback = {}
            self.on_batch_k_examples(batch=self.calls, logs=logs_callback)
            self._epoch_logs.append(logs_callback)

            logs_callback['time/' + self.name] = time.time() - t_start
            logs_callback['step/' + self.name] = self.calls
            logs_callback['examples_seen/' + self.name] = self.examples_seen

            logs.update(logs_callback)

            self.calls += 1

            if self.examples_seen_since_last_call >= self.threshold:
                self.examples_seen_since_last_call = 0

    def on_batch_end(self, batch, logs=None):
        if self.call_on_batch_begin is False:
            self._call(batch, logs)
            # There is alig here

    def on_batch_begin(self, batch, logs=None):
        if self.call_on_batch_begin is True:
            self._call(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        # Small hacky code which collects data from batch
        logs_avg = defaultdict(float)

        for logs_batch in self._epoch_logs:
            for k in logs_batch:
                logs_avg[k] += logs_batch[k]

        for k in logs_avg:
            logs_avg[k] /= len(self._epoch_logs)

        for k in logs_avg:
            logs[k] = logs_avg[k]

        self._epoch_logs = []


class DumpTensorboardSummaries(Callback):
    def __init__(self):
        super(DumpTensorboardSummaries, self).__init__()

    @property
    def file_writer(self):
        if not hasattr(self, '_file_writer'):
            self._file_writer = tensorflow.summary.FileWriter(
                self.save_path, flush_secs=10.)
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


@gin.configurable
class MetaSaver(Callback):
    def __init__(self):
        super(MetaSaver, self).__init__()

    def on_train_begin(self, logs=None):
        logger.info("Saving meta data information from the beginning of training")

        assert os.system("cp {} {}".format(sys.argv[0], self.save_path)) == 0, "Failed to execute cp of source script"

        utc_date = datetime.datetime.utcnow().strftime("%Y_%m_%d")

        time_start = time.time()
        cmd = "python " + " ".join(sys.argv)
        self.meta = {"cmd": cmd,
                     "save_path": self.save_path,
                     "most_recent_train_start_date": utc_date,
                     "execution_time": -time_start}

        json.dump(self.meta, open(os.path.join(self.save_path, "meta.json"), "w"), indent=4)

        # Copy gin configs used, for reference, to the save folder
        os.system("rm " + os.path.join(self.save_path, "*gin"))
        for gin_config in sys.argv[2].split(";"):
            os.system("cp {} {}".format(gin_config, self.save_path))

    def on_train_end(self, logs=None):
        self.meta['execution_time'] += time.time()
        json.dump(self.meta, open(os.path.join(self.save_path, "meta.json"), "w"), indent=4)
        os.system("touch " + os.path.join(self.save_path, "FINISHED"))
