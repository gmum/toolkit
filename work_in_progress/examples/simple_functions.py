
import sys
sys.path.append('/home/sieradzki/toolkit')


import numpy as np
import mandala as md

md.set_storage_backend('pickle')
md.set_graph_backend('iGraph')

from mandala.wrappers import mandala


@mandala()
def get_X():
    return np.arange(50).reshape(10,5)

@mandala()
def get_row(X, r):
    return X[r]


X = get_X()
row = get_row(X, 3)
