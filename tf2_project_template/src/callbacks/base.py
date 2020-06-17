# -*- coding: utf-8 -*-
"""
Base callback implementation. Inspired by Keras.
"""

import datetime
import json
import logging
import os
import pickle
import sys
import time
import gin
import numpy as np
import tensorflow
import copy
from tensorflow.keras import backend

from collections import defaultdict

from src import RESULTS_DIR
try:
    import neptune
except ImportError:
    pass

from src.utils import save_model, get_neptune_exp, configure_neptune_exp


logger = logging.getLogger(__name__)


class Callback(object):
    def __init__(self):
        pass

    def set_config(self, config):
        self.config = config

    def set_datasets(self, datasets):
        self.datasets = datasets

    def set_seed(self, seed):
        self.seed = seed

    def set_save_path(self, save_path):
        self.save_path = save_path

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_model(self, model):
        self.model = model

    def set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def set_params(self, params):
        self.params = params

    def get_config(self):
        return self.config

    def get_datasets(self):
        return self.datasets

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

    def __getstate__(self):
        state = self.__dict__.copy()
        if "datasets" in state:
            del state['datasets']
        if "callbacks" in state:
            del state['callbacks']
        if "cache" in state:
            del state['cache']
        for k in list(state):
            if isinstance(state[k], np.ndarray) or hasattr(state[k], '__call__') or hasattr(state[k], '__iter__'):
                del state[k]
        return state


class EvaluateCopyTask(Callback):

    def _eval_on(self, ds, samples):
        seen = 0
        correct = 0
        N = 0
        for id, (x, y) in enumerate(ds):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).to(DEVICE)
            assert len(x) == len(y)
            seen += len(x)
            y_pred = self.model(x)
            y_pred = torch.stack(y_pred).detach().cpu().numpy()
            y_pred = y_pred.transpose(1, 0, 2)
            y = y[:, -10:]
            y_pred = y_pred[:, -10:]
            y_pred = y_pred.argmax(axis=2)
            if id <= 1:
                logger.info(y[0:1])
                logger.info(y_pred[0:1])
            assert y.shape[1] == y_pred.shape[1] == 10
            assert len(y) == len(y_pred)
            N += np.prod(y_pred.shape)
            correct += (y_pred.reshape(-1, ) == y.reshape(-1, )).sum()

            if seen >= samples:
                break

        return float(correct) / N

    def on_epoch_end(self, epoch, logs):
        logs['copy_task/test_acc'] = self._eval_on(self.datasets[0][1], samples=5000)
        logs['copy_task/train_acc'] = self._eval_on(self.datasets[0][0], samples=5000)


class MeasureTime(Callback):
    def __init__(self):
        self.train_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_train_begin(self, logs=None):
        self.train_start = time.time()

    def on_batch_begin(self, batch, logs=None):
        self.batch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs['t_epoch'] = np.float64(time.time() - self.epoch_start)

    def on_batch_end(self, batch, logs=None):
        logs['t_batch'] = np.float64(time.time() - self.batch_start)


class BaseLogger(Callback):
    """Callback that accumulates epoch averages."""

    def __init__(self):
        super(BaseLogger, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = defaultdict(list)

    def on_batch_end(self, batch, logs=None):
        if logs is not None:
            for k, v in logs.items():
                self.totals[k].append(v)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.totals:
                if not k.startswith("size:"):
                    # This is a simplification if the final batch is smaller
                    logs[k] = np.mean(self.totals[k], axis=0)  # Keeps ndim > 1
                else:
                    logs[k] = np.sum(self.totals[k])


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

        if hasattr(self.optimizer, 'param_groups'):
            for group in self.optimizer.param_groups:
                group['lr'] = v * self.base_lr
            logger.info("Set learning rate to {}".format(v * self.base_lr))
        else:
            logger.info("Set learning rate to {}".format(v * self.base_lr))
            backend.set_value(self.optimizer.lr, v * self.base_lr)


@gin.configurable
class BatchLRSchedule(Callback):
    def __init__(self, base_lr, schedule):
        self.schedule = schedule
        self.base_lr = base_lr
        self.examples_seen = 0
        super(BatchLRSchedule, self).__init__()

    def on_batch_begin(self, batch, logs):
        epoch = float(self.examples_seen) / self.datasets[0][-1]['n_examples_train']

        # Epochs starts from 0
        for e, v in self.schedule:
            if epoch < e:
                break

        if hasattr(self.optimizer, 'param_groups'):
            for group in self.optimizer.param_groups:
                group['lr'] = v * self.base_lr
        else:
            backend.set_value(self.optimizer.lr, v * self.base_lr)

        logger.info("Set learning rate to {} examples seen {}".format(v * self.base_lr, epoch))

        self.examples_seen += logs['size:0']


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
            assert "size:0" in logs
            self.examples_seen += logs['size:0']
            logs['examples_seen'] = self.examples_seen
            self.examples_seen_since_last_population += logs['size:0']

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
                        save_model(self.model, self.optimizer, self.filepath)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, self.filepath))
                    save_model(self.model, self.optimizer, self.filepath)


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
        self._epoch_logs = []
        self.calls = 0
        self.frequency = frequency

    def on_batch_k_examples(self, batch, logs):
        raise NotImplementedError()

    def on_train_begin(self, logs):
        # Size of epoch of this callback is a sum of all dataset sizes
        # self.epoch_size = sum([len(d[-1]['x_train_raw']) for d in self.datasets])
        if 'x_train_raw' in self.datasets[0][-1]:
            self.epoch_size = sum([len(d[-1]['x_train_raw']) for d in self.datasets])
        else:
            self.epoch_size = sum([d[-1]['n_examples_train'] for d in self.datasets])

        if self.frequency.endswith("ex"):
            self.threshold = int(self.frequency[0:-2])
        elif self.frequency.endswith("exp"):
            self.threshold = int(self.epoch_size * float(self.frequency[0:-3]))
        else:
            raise NotImplementedError()

    def _call(self, batch, logs=None):

        if self.threshold < 0:
            return

        assert "size:0" in logs
        self.examples_seen += logs['size:0']
        self.examples_seen_since_last_call += logs['size:0']

        # Always call on batch 0
        if (self.calls == 0) or (self.examples_seen_since_last_call >= self.threshold):
            t_start = time.time()

            logs_callback = {}
            logs_callback['examples_seen/' + self.name] = self.examples_seen
            self.on_batch_k_examples(batch=self.calls, logs=logs_callback)
            self._epoch_logs.append(logs_callback)

            logger.info(
                f"Call {self.name} after or before seeing {self.examples_seen_since_last_call}, total={self.examples_seen}, threshold={self.threshold}")

            logs_callback['time/' + self.name] = time.time() - t_start
            logs_callback['step/' + self.name] = self.calls

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


class Heartbeat(Callback):
    def __init__(self, interval=10):
        self.last_time = time.time()
        self.interval = interval

    def on_train_begin(self, logs=None):
        logger.info("HEARTBEAT - train begin")
        os.system("touch " + os.path.join(self.save_path, "HEARTBEAT"))

    def on_batch_begin(self, batch, logs=None):
        if time.time() - self.last_time > self.interval:
            logger.info("HEARTBEAT")
            os.system("touch " + os.path.join(self.save_path, "HEARTBEAT"))
            self.last_time = time.time()


class DumpTensorboardSummaries(Callback):
    def __init__(self):
        super(DumpTensorboardSummaries, self).__init__()

    @property
    def file_writer(self):
        try:
            if not hasattr(self, '_file_writer'):
                self._file_writer = tensorflow.summary.create_file_writer(self.save_path)
            return self._file_writer
        except:
            logger.warning("Failed to generate tensorboard summaries")

    def on_epoch_end(self, epoch, logs=None):
        try:
            with self.file_writer.as_default():
                for key, value in logs.items():
                    value = float(value)
                    tensorflow.summary.scalar(key, value, step=epoch)
        except:
            logger.warning("Failed to generate tensorboard summaries")


@gin.configurable
class WeightNorm(CallbackPickableEveryKExamples):
    def __init__(self, frequency="1.0exp"):
        self.it = 0
        super(WeightNorm, self).__init__(name='weight_norm', frequency=frequency, on_batch_begin=False)

    def on_batch_k_examples(self, batch, logs):

        if isinstance(self.model, tensorflow.keras.Model):

            # WARNING: Could be a bit slow for some models
            for l in self.model.layers:
                for w in l.trainable_weights:
                    logs['weight_norm/' + l.name + "_" + w.name] = (np.linalg.norm(w.numpy()))

        else:
            logger.warning("I don't know how to measure weight norms of pytorch model")


@gin.configurable
class SaveWeights(CallbackPickableEveryKExamples):
    def __init__(self, frequency="1.0exp"):
        self.it = 0
        super(SaveWeights, self).__init__(name='save_state', frequency=frequency, on_batch_begin=False)

    def on_batch_k_examples(self, batch, logs):
        save_model(model=self.model, optimizer=self.optimizer,
                   filename=os.path.join(self.save_path, f"model_it{self.it}"))
        self.it += 1

@gin.configurable
class NeptuneMonitor(Callback):
    """
    Callback that writes to Neptune.

    Notes
    -----
    Requires the following env variables: NEPTUNE_TOKEN, NEPTUNE_PROJECT_NAME, NEPTUNE_USER_NAME
    """
    def __init__(self):
        super(NeptuneMonitor, self).__init__()

    def on_train_begin(self, logs):
        neptune_name = self.save_path
        if neptune_name.startswith(RESULTS_DIR):
            neptune_name = neptune_name[len(RESULTS_DIR):]
        configure_neptune_exp(self.save_path)

    def on_epoch_end(self, epoch, logs=None):
        neptune_exp = get_neptune_exp()
        for k in logs:
            neptune_exp.send_metric(k, logs[k])


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
            os.system("cp {} {}/base_config.gin".format(gin_config, self.save_path))

        with open(os.path.join(self.save_path, "config.gin"), "w") as f:
            f.write(gin.operative_config_str())

    def on_train_end(self, logs=None):
        self.meta['execution_time'] += time.time()
        json.dump(self.meta, open(os.path.join(self.save_path, "meta.json"), "w"), indent=4)
        os.system("touch " + os.path.join(self.save_path, "FINISHED"))
