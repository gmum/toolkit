"""
Minor utilities
"""

import sys
from functools import reduce

import traceback
import logging
import argparse
import optparse
import datetime
import sys
import pprint
import types
import time
import copy
import subprocess
import glob
from collections import OrderedDict
import os
import signal
import atexit
import json
import inspect

from logging import handlers

import argh
import gin
from gin.config import _OPERATIVE_CONFIG

import torch
from torch.nn.modules.module import _addindent

logger = logging.getLogger(__name__)

def acc(y_pred, y_true):
    _, y_pred = y_pred.max(1)
    # _, y_true = y_true.max(1)
    acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100

def save_weights(model, optimizer, filename):
    """
    Save all weights necessary to resume training
    """
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)

from contextlib import contextmanager


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


def summary(model, file=sys.stderr):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stderr:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        print(string, file=file)
    return count