# -*- coding: utf-8 -*-
"""
 Wraps a python script as cmdline tool

 Similar to https://github.com/IDSIA/sacred and
 https://github.com/micheles/plac/blob/0.9.6/doc/plac.pdf
"""
__author__ = "Stanislaw 'kudkudak' Jastrzebski"

import logging

logger = logging.getLogger(__name__)

import datetime
import inspect
import json
import logging
import sys
import optparse
import os
import time
import traceback
import base64

from six import string_types
try:
    import dropbox
except ImportError:
    logging.warning("Failed to import Dropbox. Uploading results won't work.")

import os

def print_as_json(d):
    print(json.dumps(d, sort_keys=True, indent=4))


def json_from_str(x):
    """
    Basically hopes string is a correct json, but if it is base64 or empty string also works
    """
    if len(x) == 0 and isinstance(x, string_types):
        return ""
    elif isinstance(x, string_types):
        try:
            logger.info("Trying to json decode via base64 " + x)
            return json.loads(base64.b64decode(x))
        except:
            logger.info("Trying to decode " + x)
            return json.loads(x)

    else:
        raise ValueError("Failed parsing " + str(x))


def pprint_dict(d):
    return json.dumps(d, sort_keys=True, indent=4)


def _get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    adopted from http://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    if defaults is None or len(defaults) == 0:
        return {}
    return dict(zip(args[-len(defaults):], defaults))


def utc_date(format="%Y_%m_%d"):
    return datetime.datetime.utcnow().strftime(format)


def utc_timestamp():
    return str(int(10 * (datetime.datetime.utcnow() - datetime.datetime(1970, 1,
                                                                        1)).total_seconds()))


def timestamp_namer(fnc_kwargs=None):
    return utc_date() + "_" + utc_timestamp()


from collections import OrderedDict


def catch_run_name_namer(fnc_kwargs):
    if "run_name" not in fnc_kwargs or not fnc_kwargs['run_name'] or len(fnc_kwargs['run_name']) == 0:
        raise ValueError("Expected passed run_name")
    return fnc_kwargs['run_name']


def kwargs_namer(fnc_kwargs=None):
    return "_".join("{}={}".format(k, v) for k, v in OrderedDict(fnc_kwargs).iteritems()
                        if k not in ['_time_last_log', '_meta', 'base_fname', 'save_to', 'reload'])


def construct_kwargs_namer(ignore_keys=[], hash_it=False):
    def _kwargs_namer(fnc_kwargs=None):
        name = "_".join("{}={}".format(k, v) for k, v in OrderedDict(fnc_kwargs).iteritems()
                            if k not in (ignore_keys + ['_time_last_log', '_meta', 'base_fname', 'save_to', 'reload']))
        hash_name = str(hash(name))
        if hash_name[0] == "-":
            hash_name = "9" + hash_name[1:]
        return hash_name if hash_it else name

    return _kwargs_namer

def guess_params_mode():
    if len(sys.argv) == 1 or sys.argv[1][0] == "-":
        return "shell"
    else:
        return "json"


def wrap(fnc, save_root_dir=".", base_fname="", save_results=True,
        save_logs=False, namer=timestamp_namer, ignore_args=[]):
    """
    Params
    -----
    save_root_dir: str
        puts log and json file into save_root_dir/base_fname[.log/.json] if base_fname is relative
    else into base_fname[.log/.json]
    """

    params_mode = guess_params_mode()

    ### 1. Populate OptionParser (with special arguments) ###
    arg_types = {}
    parser = optparse.OptionParser()
    defaults = _get_default_args(fnc)
    file_name = inspect.getmodule(inspect.stack()[1][0]).__file__
    defaults['base_fname'] = base_fname
    del base_fname

    fnc_arg_spec = inspect.getargspec(fnc)[0]

    for varname, default_value in defaults.items():
        if default_value is not None:
            arg_types[varname] = type(default_value)

    if params_mode == "shell":
        for varname in fnc_arg_spec:
            if varname in ['base_fname', '_meta', '-', '_time_last_log'] + ignore_args:
                continue
            parser.add_option("--" + varname, type="str" if arg_types.get(varname, "str") == "json"
            else arg_types.get(varname, "str"), default=defaults.get(varname, None))

        parser.add_option("--base_fname", type="str", default=defaults['base_fname'])
        parser.add_option("--save_logs", type="int", default=save_logs)
        parser.add_option("--save_results", type="int", default=save_results)

    ### 2. Run cmd ###
    if params_mode == "shell":
        (opts, args) = parser.parse_args()
        if len(args) > 0:
            raise RuntimeError("Positional arguments are prohibited, terminating.")
        args = opts.__dict__
    elif params_mode == "json":
        logging.info("Interpreting {} as config".format(sys.argv[1]))
        args = dict(defaults)
        args.update(json.load(open(sys.argv[1])))
        args['save_logs'] = save_logs
        args['save_results'] = save_results
    else:
        raise RuntimeError("params_mode " + params_mode + " not supported.")

    if "save_root_dir" in args:
        save_root_dir = args["save_root_dir"]

    ### 2.5. Create save_root_dir ###
    if not (os.path.isabs(args['base_fname']) or os.path.exists(save_root_dir)):
        logging.info("Creating folder " + save_root_dir)
        os.system("mkdir -p " + save_root_dir)

    ### 3. Configure ###
    time_start = time.time()
    cmd = "{} {}".format(file_name, " ".join(
        "--{} {}".format(k, v) for k, v in args.items() if k in fnc_arg_spec))
    meta = {"base_fname": args['base_fname'],
        "cmd": cmd,
        "args": dict(args),
        "start": utc_date(),
        "time": time.time() - time_start}

    # Add some special args if user wants to seem them
    args['base_fname'] = args['base_fname'] if len(args['base_fname'].strip()) > 0 \
        else os.path.join(save_root_dir, namer(fnc_kwargs=args))
    args['_meta'] = meta

    if not os.path.isabs(args['base_fname']):
        output_fname, log_fname = os.path.join(save_root_dir, args['base_fname'], "run.json"), \
            os.path.join(save_root_dir, args['base_fname'], "run.log")
    else:
        output_fname, log_fname = args['base_fname'] + "/run.json", args['base_fname'] + "/run.log"

    if not os.path.exists(os.path.dirname(log_fname)):
        logger.info("Creating dir " + os.path.dirname(log_fname))
        os.system("mkdir -p " + os.path.dirname(log_fname))

    args['_time_last_log'] = os.path.getmtime(log_fname) if os.path.exists(log_fname) else -1

    if save_logs:
        fh = logging.FileHandler(log_fname, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger('').addHandler(fh)  # Assumes script logs using logging

    logger.info("Base fname={} @ {}".format(args['base_fname'], utc_date()))
    save_logs = args['save_logs']
    save_results = args['save_results']

    fnc_args = dict(args)
    for k in args:
        if k not in fnc_arg_spec:
            del fnc_args[k]

    ### 4. Check configuration ###
    for varname in fnc_arg_spec:
        if fnc_args[varname] is None:
            raise RuntimeError("Not passed required " + varname)

    # 2b. Sanity checks and conversions on arg
    for key, val in args.items():
        try:
            if arg_types.get(key, "str") == "json" and isinstance(val,
                                                                  string_types):
                args[key] = json.loads(val)
        except Exception as e:
            logger.error(
                "Failed parsing key {} with val {} to json, typeof val is {}".format(
                    key, val, type(val)))
            raise e

    ### 5. Run function ###
    logger.info("Running " + cmd)

    try:
        result = fnc(**fnc_args)
        logging.info("Finished " + args['base_fname'])
    except KeyboardInterrupt:
        if (not save_results and not save_logs) or raw_input("Save meta/logs? Y/N") == "N":
            exit(0)
        result = None

    meta['time'] = time.time() - time_start
    to_save = {"meta": meta}

    if save_results:
        to_save["result"] = result
    to_save["config"] = fnc_args

    with open(output_fname, "w") as f:
        f.write(
            json.dumps(to_save, sort_keys=True, indent=4))

    return result


def ensure_is_local(path, prefix):
    if os.path.isabs(path):
        return path
    else:
        if not path.startswith(prefix):
            raise Exception("Unexpected {} not starting with {}".format(path, prefix))
        return path[len(prefix):]

def dropbox_dir_download(directory, target_dir, app_key=os.environ.get("BURRITO_APP_KEY", None),
        app_secret=os.environ.get("BURRITO_APP_SECRET", None),
        access_token=os.environ.get("BURRITO_ACCESS_TOKEN", None)):
    """
    Downloads given folder to local drive
    """

    flow = dropbox.client.DropboxOAuth2FlowNoRedirect(app_key, app_secret)
    authorize_url = flow.start()
    try:
        client = dropbox.client.DropboxClient(access_token)
    except:
        print '1. Go to: ' + authorize_url
        print '2. Click "Allow" (you might have to log in first)'
        print '3. Copy the authorization code.'
        code = raw_input("Enter the authorization code here: ").strip()
        access_token, user_id = flow.finish(code)
        client = dropbox.client.DropboxClient(access_token)
        print 'linked account: ', client.account_info()

    files = map(lambda x: x['path'], client.metadata(directory)['contents'])

    if not os.path.exists(target_dir):
        os.system("mkdirs -p " + target_dir)

    for f in files:
        logger.info("Downloading " + f)
        z = client.get_file(f)
        with open(os.path.join(target_dir, os.path.basename(f)), "wb") as f_target:
            f_target.write(z.read())

def upload_files_to_dropbox(directory, F,
        app_key=os.environ.get("BURRITO_APP_KEY", None),
        app_secret=os.environ.get("BURRITO_APP_SECRET", None),
        access_token=os.environ.get("BURRITO_ACCESS_TOKEN", None)):
    """
    Uploads results to Dropbox using registered folder-app

    Example
    -------
    >> P = {"n_batches": 10, "lr": 0.01}
    >> upload_experiment_results_to_dropbox("my_experiment/", {"params.json": StringIO.StringIO(json.dumps(P, indent=2))})
    """
    flow = dropbox.client.DropboxOAuth2FlowNoRedirect(app_key, app_secret)
    authorize_url = flow.start()
    try:
        client = dropbox.client.DropboxClient(access_token)
    except:
        print '1. Go to: ' + authorize_url
        print '2. Click "Allow" (you might have to log in first)'
        print '3. Copy the authorization code.'
        code = raw_input("Enter the authorization code here: ").strip()
        access_token, user_id = flow.finish(code)
        client = dropbox.client.DropboxClient(access_token)
        print 'linked account: ', client.account_info()

    for f_name in F:
        response = client.put_file(os.path.join(directory, f_name), F[f_name])
        print "uploaded:", response
