#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import time

import numpy as np

from coonhound import DirectoryHound
from storage import DirStorage, with_storage


@with_storage
def hstack(a, b):
    return np.hstack((a,b))

@with_storage
def get_X():
    print "sleep 5"
    time.sleep(5)
    print "woke up"
    return np.arange(50).reshape(10,5)

@with_storage
def get_Y():
    return np.arange(50).reshape(10,5)

@with_storage
def get_row(X, r):
    return X[r]

@with_storage
def dummy(X, Y):
    return X


hound = DirectoryHound(".")
def DS(**kwargs):
    return DirStorage(hound.find(**kwargs))


get_X(input={}, output=(DS(name="tables", which="X", favourite_number=43),))
