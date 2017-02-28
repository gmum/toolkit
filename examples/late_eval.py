
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

@mandala(store=False)
def get_row(X, r):
    return X[r]

@mandala(store=False)
def sum_row(row):
    return np.sum(row)

X = get_X()
r = 3
row = get_row(X, r)
#row_sum = sum_row(row)

print "Sum for row {} is {}".format(r, row.eval())