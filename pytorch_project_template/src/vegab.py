# -*- coding: utf-8 -*-
"""
Main function that can be used to standardize training and evaluation scripts. Named
after vegetarian Kebap place in Cracow, Poland. Shipped as single file for now.

* Called function accepts:
- config dictionary (can be name from registry, or just json)
- save path

* Main function:
- Redirects all stdout and stderr to file. Configures saving logger to file
- Checks validity of arguments
- Supports arguments for customizing save location: default_folder, and default_naming, prefix
- Supports additional scenarios for saving: If KeyboardInterrupted: asks if save results, if debug flag
removes everything after run finishes or throws error

* Additional plugins (plugin is an object supporting events):
- VisdomPlotter
- MetaSaver
        a) time running (meta['run_time'])
        b) host (meta['host'])
        c) date of execution (meta['date'])
        e) code of script
        f) cmd used to run (meta['cmd'])
        g) config.json
        h) script source code


Acknoledgments
Inspired by Dzimitry Bahdanau code, burrito (internally used in GMUM) and argh
"""
from __future__ import print_function

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

logger = logging.getLogger(__name__)


### Config registry ###

class ConfigRegistry(object):
    def __init__(self):
        self._configs = {}

    def get_root_config(self):
        return self['root']

    def set_root_config(self, config):
        self['root'] = config

    def __getitem__(self, name):
        # copy on read
        return copy.deepcopy(self._configs[name])

    def __setitem__(self, name, config):
        if name in self._configs:
            raise KeyError("Config already registered " + name)
        self._configs[name] = copy.deepcopy(config)


def config_from_folder(path):
    """
    Returns
    -------
    c: ConfigRegistry
        ConfigRegistry constructed from jsons in path
    """

    files = glob.glob(os.path.join(path, "*json"))
    c = ConfigRegistry()

    for f in files:
        c[os.path.splitext(os.path.basename(f))[0]] = json.load(open(f))

    if 'root' not in c:
        raise IOError("Didn't find root config in config registry")

    return c


### Simple plugin system ###

def _utc_date(format="%Y_%m_%d"):
    return datetime.datetime.utcnow().strftime(format)


def _utc_timestamp():
    return str(int(10 * (datetime.datetime.utcnow() - datetime.datetime(1970, 1,
        1)).total_seconds()))


def _timestamp_namer(fnc_kwargs=None):
    return _utc_date() + "_" + _utc_timestamp()


def _kwargs_namer(fnc_kwargs=None, args_to_use=set()):
    if len(args_to_use):
        return "_".join("{}={}".format(k, v) for k, v in OrderedDict(fnc_kwargs).iteritems()
            if k not in ['config', 'save_path'])
    else:
        return "_".join("{}={}".format(k, v) for k, v in OrderedDict(fnc_kwargs).iteritems()
            if k in args_to_use)


def _time_prefix(fnc_kwargs=None):
    return "_".join("{}={}".format(k, v) for k, v in OrderedDict(fnc_kwargs).iteritems()
        if k not in ['config', 'save_path'])


class VegabPlugin(object):
    def on_parsed_args(self, args):
        return args

    def on_before_call(self, config, save_path):
        pass

    def on_after_call(self, config, save_path):
        pass


class VisdomPlotter(VegabPlugin):
    """
    Starts visdom_plotter.py which sends all info to visdom server
    """

    def on_before_call(self, config, save_path):
        if not "VISDOM_SERVER" in os.environ:
            raise Exception("Please make sure there is env variable VISDOM_SERVER")
        ret = subprocess.Popen([os.path.join(os.path.dirname(__file__), "../visdom_plotter.py"),
            "--visdom-server={}".format(os.environ['VISDOM_SERVER']), "--folder={}".format(save_path)])
        time.sleep(0.1)
        if ret.returncode is not None:
            raise Exception()
        atexit.register(lambda: os.kill(ret.pid, signal.SIGINT))


class AutomaticNamer(VegabPlugin):
    """
    Adds prefix or decides on full name
    """

    def __init__(self, as_prefix=True, namer=""):
        self.as_prefix = as_prefix

        if isinstance(namer, types.FunctionType):
            self.namer = namer
        else:
            # Fine to have ifs here as long as we have just few choices
            if namer == "kwargs_namer":
                self.namer = _kwargs_namer
            if namer == "timestamp_namer":
                self.namer = _timestamp_namer
            else:
                raise NotImplementedError("Not implemented namer " + namer)

    def on_parsed_args(self, args):
        if not os.path.exists(args.save_path):
            name = self.namer(args.__dict__)

            if self.as_prefix:
                args.save_path = os.path.join(os.path.dirname(args.save_path),
                    name + "_" + os.path.basename(args.save_path))
            else:
                args.save_path = os.path.join(args.save_path, name)

        logger.info("Saving to " + args.save_path)

        return args


class TBLinker(VegabPlugin):
    """
    Saves useful meta information (including source code)
    """

    def __init__(self, tb_dir):
        self.tb_dir = tb_dir

    def on_before_call(self, config, save_path):
        os.system("ln -s " + save_path + " " + os.path.join(self.tb_dir))

    def on_after_call(self, config, save_path):
        os.system("rm " + os.path.join(self.tb_dir, os.path.basename(save_path)))


class MetaSaver(VegabPlugin):
    """
    Saves useful meta information (including source code)
    """

    def on_before_call(self, config, save_path):
        frame = inspect.stack()[1 + 1]  # Hardcoded, change if u plugin calling is change
        module = inspect.getmodule(frame[0])

        f_name = module.__file__
        assert os.system("cp {} {}".format(f_name, save_path)) == 0, "Failed to execute cp of source script"

        time_start = time.time()
        cmd = "python " + " ".join(sys.argv)
        self.meta = {"cmd": cmd,
            "save_path": save_path,
            "start_utc_date": _utc_date(),
            "execution_time": -time_start}

        json.dump(config, open(os.path.join(save_path, "config.json"), "w"), indent=4)
        json.dump(self.meta, open(os.path.join(save_path, "meta.json"), "w"), indent=4)

    def on_after_call(self, config, save_path):
        self.meta['execution_time'] += time.time()
        json.dump(self.meta, open(os.path.join(save_path, "meta.json"), "w"), indent=4)


### Redirect code ###
# Stolen from Dzimitry Bahdanau redirecting code,
# works more reliably than previous version

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


def _get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    adopted from http://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    if defaults is None or len(defaults) == 0:
        return {}
    return dict(zip(args[-len(defaults):], defaults))


def _add_config_arguments(config, parser):
    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(
                "--" + key, dest=key, default=None, action="store_true",
                help="Enable a setting from the configuration")
            parser.add_argument(
                "--no_" + key, dest=key, default=None, action="store_false",
                help="Disable a setting from the configuration")
        else:
            convertor = type(value)
            # let's assume all the lists in our configurations will be
            # lists of ints
            if isinstance(value, list) or isinstance(value, list):
                convertor = lambda s: eval(s)
                # convertor = lambda s: map(int, s.split(','))
            parser.add_argument(
                "--" + key, type=convertor,
                help="A setting from the configuration")


def _add_arguments_from_func_signature(func, parser):
    defaults = _get_default_args(func)
    for varname, default_value in defaults.items():
        if default_value is None:
            raise NotImplementedError("Vegab works only for functions with fully specified default values")
    _add_config_arguments(defaults, parser)


def parse_logging_level(logging_level):
    """
    :param logging_level: Logging level as string
    :return: Parsed logging level
    """
    lowercase = logging_level.lower()
    if lowercase == 'debug': return logging.DEBUG
    if lowercase == 'info': return logging.INFO
    if lowercase == 'warning': return logging.WARNING
    if lowercase == 'error': return logging.ERROR
    if lowercase == 'critical': return logging.CRITICAL
    raise ValueError('Logging level {} could not be parsed.'.format(logging_level))

from logging import handlers

def configure_logger(name=__name__,
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

    if len(logging.getLogger(name).handlers) != 0:
        raise Exception("Please pass unconfigured logger")

    if console_logging_level is None and file_logging_level is None:
        return  # no logging

    if isinstance(console_logging_level, str):
        console_logging_level = parse_logging_level(console_logging_level)

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


### Main driver ###

def wrap_no_config_registry(func, plugins=[]):
    configure_logger('', log_file=None)

    # Create parser and get config
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", help="The destination for saving")
    _add_arguments_from_func_signature(func, parser)

    args = parser.parse_args()

    # Plugin order matters sometimes (just like callbacks order in Blocks or Keras)
    for p in plugins:
        args = p.on_parsed_args(args)

    # Do some configurations, then run with redirection
    def call_training_func():
        logger.info("Calling function {}".format(func.__name__))

        if not os.path.exists(args.__dict__['save_path']):
            os.system("mkdir -p " + args.__dict__['save_path'])

        func(**args.__dict__)

        logger.info("Finished {}".format(func.__name__))

        os.system("touch " + os.path.join(args.__dict__['save_path'], "FINISHED"))

        for p in plugins:
            p.on_after_call(args.__dict__, args.save_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    for p in plugins:
        p.on_before_call(args.__dict__, args.save_path)

    run_with_redirection(
        os.path.join(args.save_path, 'stdout.txt'),
        os.path.join(args.save_path, 'stderr.txt'),
        call_training_func)()

from collections import defaultdict

def _parse_args(config_registry):
    # Returns parsed args (using heavily Argument Parser)

    # Create parser and get config
    parser = argparse.ArgumentParser("vegab-wrapped script")
    parser.add_argument("config", help="The configuration")
    parser.add_argument("save_path", help="The destination for saving")
    if isinstance(config_registry, str):
        config_registry = config_from_folder(config_registry)
    elif isinstance(config_registry, ConfigRegistry):
        pass
    else:
        raise NotImplementedError("Not understood config registry")
    _add_config_arguments(config_registry.get_root_config(), parser)

    # Parse some dict kwargs. Hacky, but contained.
    arg_kwargs = defaultdict(dict)
    cleared_args = []
    for arg in sys.argv[1:]:
        if "." in arg:
            if "=" not in arg:
                raise NotImplementedError("Please pass dict arguments as --model.beta=1, not --model.beta 1")
            arg = arg[2:] # Remove '--'
            argnamekey, val = arg.split("=")
            argname, key = argnamekey.split(".")
            arg_kwargs[argname + "_kw"][key] = float(val)
        else:
            cleared_args.append(arg)

    # Rest is parsed using the parses
    d = parser.parse_args(cleared_args).__dict__

    # TODO: Add some checking?
    print(arg_kwargs)
    d.update(arg_kwargs)
    return d

def wrap(config_registry, func, plugins=[MetaSaver()], **training_func_kwargs):
    configure_logger('', log_file=None)

    args = _parse_args(config_registry)

    # Plugin order matters sometimes (just like callbacks order in Blocks or Keras)
    for p in plugins:
        args = p.on_parsed_args(args)

    config = config_registry[args['config']]
    for key in args:
        if key not in config.keys():
            if key.endswith("kw"): # Special treatment
                config[key] = args[key]
            elif key not in ['save_path', 'config']:
                raise Exception("Not recognised " + key)
        elif args[key] is not None:
            config[key] = args[key]

    # Do some configurations, then run with redirection
    def call_training_func():
        pprint.pprint(config)
        logger.info("Calling function {}".format(func.__name__))
        func(config, args['save_path'], **training_func_kwargs)
        logger.info("Finished {}".format(func.__name__))
        os.system("touch " + os.path.join(args['save_path'], "FINISHED"))
        for p in plugins:
            p.on_after_call(config, args['save_path'])

    if not os.path.exists(args['save_path']):
        os.mkdir(args['save_path'])

    for p in plugins:
        p.on_before_call(config, args['save_path'])

    run_with_redirection(
        os.path.join(args['save_path'], 'stdout.txt'),
        os.path.join(args['save_path'], 'stderr.txt'),
        call_training_func)()
