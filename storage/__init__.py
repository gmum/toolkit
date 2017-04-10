#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import json
import json
import os
import cPickle as pickle

import h5py
import numpy as np
import pandas as pd
import scipy


class OverwriteAttempt(Exception):
    pass


class Storage(object):

    def __init__(self):

        self._dirname = None
        self._filename = None
        self._meta_filename = None

        # format-specific
        self._pandas_h5_key = "data"
        self._numpy_h5_key = "data"

    def _save_pandas(self, data):
        data.to_hdf(self._filename, self._pandas_h5_key)

    def _load_pandas(self):
        return data.to_hdf(self._filename, self._pandas_h5_key)

    def _save_numpy(self, data):
        with h5py.File(self._filename, 'w') as h5f:
            h5f.create_dataset(self._numpy_h5_key, data=data)

    def _load_numpy(self):
        with h5py.File(self._filename, 'r') as h5f:
            return h5f[self._numpy_h5_key][:]

    def _save_other(self, data):
        with open(self._filename, 'w') as f_out:
            pickle.dump(data, f_out)

    def _load_other(self):
        with open(self._filename, 'r') as f_in:
            return pickle.load(f_in)

    def _load_meta(self):
        with open(self._meta_filename, 'r') as f_in:
            return json.load(f_in)

    def _save_meta(self, metadata):
        s = json.dumps(metadata, sort_keys=True)
        with open(self._meta_filename, 'w') as f_out:
            f_out.write(s)

    def _update_meta(self, key, value):
        meta = self._load_meta()
        meta[key] = value
        self._save_meta(meta)

    def _read_meta(self, key):
        return self._load_meta()[key]

    def _get_data_type(self, data):
        if isinstance(data, (pd.Series, pd.DataFrame, pd.SparseSeries, pd.SparseDataFrame)):
            return "pandas"
        elif isinstance(data, np.ndarray):
            return "numpy"
        else:
            return "other"

    def _get_saver(self, data_type):
        return {
            "pandas": self._save_pandas,
            "numpy":  self._save_numpy,
            "other":  self._save_other,
        }[data_type]

    def _get_loader(self, data_type):
        return {
            "pandas": self._load_pandas,
            "numpy":  self._load_numpy,
            "other":  self._load_other,
        }[data_type]

    def _require_meta(self):
        if not os.path.isfile(self._meta_filename):
            self._save_meta({})

    def _require_dir(self):
        if os.path.exists(self._dirname):
            if os.path.isfile(self._dirname):
                raise IOError("Cannot create directory, " + self._dirname + " already exists, but is a file.")
        else:
            os.makedirs(self._dirname)

    def save(self, data):
        if self.exists():
            raise OverwriteAttempt("Filename: " + self._filename + " already exists, aborting.")
        data_type = self._get_data_type(data)
        self._get_saver(data_type)(data)
        self._update_meta("data_type", data_type)

    def load(self):
        return self._get_loader(self._read_meta("data_type"))()

    def exists(self):
        return os.path.isfile(self._filename)


class DirStorage(Storage):
    """ Store in given directory, guess type, choose extension, add metadata etc. """
    def __init__(self, path):
        super(DirStorage, self).__init__()
        self._dirname = path
        self._filename = os.path.join(path, "data")
        self._meta_filename = os.path.join(path, "meta.json")
        self._require_dir()
        self._require_meta()


def with_storage(function):

    def wrapped(input={}, output=[]):
        assert len(output) > 0, "Either you forgot to specify 'output' or you do not want to use this wrapper..."
        assert isinstance(input, dict)
        assert isinstance(output, (list, tuple))

        output_exists = []
        nb_none = 0
        for x in output:
            if isinstance(x, Storage):
                output_exists.append(x.exists())
            elif x is None:
                output_exists.append(False)
                nb_none += 1
            else:
                raise ValueError("'output' must be a list of Storage instances or None values.")

        if (np.array(output_exists) == True).all() == True:
            output_values = [x.load() for x in output]
            return tuple(output_values) if len(output_values) > 1 else output_values[0]

        else:
            input_kwargs = {}
            for key, value in input.iteritems():
                input_kwargs[key] = value if not isinstance(value, Storage) else value.load()
            output_values = function(**input_kwargs)
            if isinstance(output_values, tuple):
                assert len(output) == len(output_values), "Output has lenght " + str(len(output)) + ", function returned " + str(len(output_values)) + " values."
                for s, v in zip(output, output_values):
                    if s is not None:
                        s.save(v)
                return output_values
            else:
                assert len(output) == 1, "Output has lenght " + str(len(output)) + ", function returned 1 value."
                output[0].save(output_values)
                return output_values

    return wrapped
