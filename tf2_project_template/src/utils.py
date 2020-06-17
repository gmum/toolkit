"""
Minor utilities
"""

import csv
import h5py
import logging
import os
import sys
import argh
import gin
import copy
import tensorflow
import os

from contextlib import contextmanager
from functools import reduce
from logging import handlers
from gin.config import _CONFIG
from tensorflow import keras


custom_tf_objects = {}

logger = logging.getLogger(__name__)

_NEPTUNE = {}

try:
    import torch
    from torch import nn
    from torch.nn import Module
except ImportError:
    class Module():
        pass

try:
    import neptune
except ImportError:
    pass

def save_model(model, optimizer, filename):
    """
    Save all weights and state necessary to resume training
    """
    if isinstance(model, keras.Model):
        if not filename.endswith(".h5"):
            filename = filename + ".h5"
        tensorflow.keras.models.save_model(model, h5py.File(filename, 'w'),
                                           overwrite=True, include_optimizer=True, save_format="h5")
    else:
        raise NotImplementedError()


def restore_model(model, filename):
    if isinstance(model, keras.Model):
        if not filename.endswith(".h5"):
            filename += ".h5"
        model.load_weights(filename, by_name=True)
        return model
    else:
        raise NotImplementedError()


def restore_model_and_optimizer(model, optimizer, filename):
    if isinstance(model, keras.Model):
        if not filename.endswith(".h5"):
            filename += ".h5"
        model = keras.models.load_model(h5py.File(filename), custom_objects=custom_tf_objects)
        optimizer = model.optimizer
        return model, optimizer
    else:
        raise NotImplementedError()


class Fork(object):
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)

    def flush(self):
        self.file1.flush()
        self.file2.flush()


@contextmanager
def replace_logging_stream(file_):
    root = logging.getLogger()
    if len(root.handlers) != 1:
        print(root.handlers)
        raise ValueError("Don't know what to do with many handlers")
    if not isinstance(root.handlers[0], logging.StreamHandler):
        raise ValueError
    stream = root.handlers[0].stream
    root.handlers[0].stream = file_
    try:
        yield
    finally:
        root.handlers[0].stream = stream


@contextmanager
def replace_standard_stream(stream_name, file_):
    stream = getattr(sys, stream_name)
    setattr(sys, stream_name, file_)
    try:
        yield
    finally:
        setattr(sys, stream_name, stream)


def gin_wrap(fnc):
    def main(save_path, config, bindings=""):
        # You can pass many configs (think of them as mixins), and many bindings. Both ";" separated.
        gin.parse_config_files_and_bindings(config.split("#"), bindings.replace("#", "\n"))
        if not os.path.exists(save_path):
            logger.info("Creating folder " + save_path)
            os.system("mkdir -p " + save_path)

        run_with_redirection(os.path.join(save_path, "stdout.txt"),
                             os.path.join(save_path, "stderr.txt"),
                             fnc)(save_path)

    argh.dispatch_command(main)


def run_with_redirection(stdout_path, stderr_path, func):
    def func_wrapper(*args, **kwargs):
        with open(stdout_path, 'a', 1) as out_dst:
            with open(stderr_path, 'a', 1) as err_dst:
                out_fork = Fork(sys.stdout, out_dst)
                err_fork = Fork(sys.stderr, err_dst)
                with replace_standard_stream('stderr', err_fork):
                    with replace_standard_stream('stdout', out_fork):
                        with replace_logging_stream(err_fork):
                            func(*args, **kwargs)

    return func_wrapper


def configure_neptune_exp(name):
    global _NEPTUNE
    if 'NEPTUNE_TOKEN' not in os.environ:
        logger.warning("Neptune couldn't be configured. Couldn't find NEPTUNE_TOKEN. ")
        return

    NEPTUNE_TOKEN = os.environ['NEPTUNE_TOKEN']
    NEPTUNE_USER = os.environ['NEPTUNE_USER']
    NEPTUNE_PROJECT = os.environ['NEPTUNE_PROJECT']
    C = copy.deepcopy(_CONFIG)
    C = {k[1].split(".")[-1]: v for k, v in C.items()}  # Hacky way to simplify config
    logger.info("Initializing neptune to name " + name)
    project = neptune.Session(NEPTUNE_TOKEN).get_project(f'{NEPTUNE_USER}/{NEPTUNE_PROJECT}')
    exp = project.create_experiment(name=name, params=C)
    _NEPTUNE['default'] = exp
    logger.info("Initialized neptune")

def get_neptune_exp(name=None):
    global _NEPTUNE
    if name is not None:
        return _NEPTUNE[name]
    else:
        return _NEPTUNE['default']


def configure_logger(name='',
                     console_logging_level=logging.INFO,
                     file_logging_level=None,
                     log_file=None):
    """
    Configures logger
    :param name: logger name (default=module name, __name__)
    :param console_logging_level: level of logging to console (stdout), None = no logging
    :param file_logging_level: level of logging to log file, None = no logging
    :param log_file: path to log file (required if file_logging_level not None)
    :return instance of Logger class
    """

    if file_logging_level is None and log_file is not None:
        print("Didnt you want to pass file_logging_level?")

    if len(logging.getLogger(name).handlers) != 0:
        print("Already configured logger '{}'".format(name))
        return

    if console_logging_level is None and file_logging_level is None:
        return  # no logging

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if console_logging_level is not None:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(format)
        ch.setLevel(console_logging_level)
        logger.addHandler(ch)

    if file_logging_level is not None:
        if log_file is None:
            raise ValueError("If file logging enabled, log_file path is required")
        fh = handlers.RotatingFileHandler(log_file, maxBytes=(1048576 * 5), backupCount=7)
        fh.setFormatter(format)
        logger.addHandler(fh)

    logger.info("Logging configured!")

    return logger


def dict_to_csv(path, d):
    with open(path, 'w') as f:
        for k, v in d.items():
            f.write(f'{k}, {v}\n')


def csv_to_dict(path):
    reader = csv.reader(open(path, "r"))
    d = {}
    for k, v in reader:
        d[k] = v
    return d
