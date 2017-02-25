import sys
sys.path.append('/home/sieradzki/toolkit')

# TODO: clear all cache or set new paths in __init__ before running tests

import numpy as np
import os
import mandala as md

md.set_storage_backend('pickle')
md.set_graph_backend('iGraph')

from mandala.wrappers import mandala


@mandala()
def get_X():
    return np.arange(10).reshape(5,2)

@mandala()
def get_Y():
    return np.arange(10).reshape(5,2)

@mandala()
def hstack(x, y):
    return np.hstack([x,y])


X = get_X()
Y = get_Y()

XY = hstack(x=X, y=Y)
YX = hstack(x=Y, y=X)

assert len(os.listdir(md.STORAGE_PATH)) == 4