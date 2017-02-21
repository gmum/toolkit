
import sys
sys.path.append('/home/sieradzki/toolkit')


import numpy as np
import mandala as md
from mandala.wrappers import wrap

md.set_storage_backend('pickle')
md.graph.initialize_graph(md.GRAPH_PATH)

@wrap
def get_X():
    return np.arange(50).reshape(10,5)

@wrap
def get_Y():
    return np.arange(50).reshape(10,5)

@wrap
def get_row(X, r):
    return X[r]

@wrap
def dummy(X, Y):
    return X


node1 = get_X()
node2 = get_Y()
node3 = dummy(node1, node2)

node4 = get_row(node1, 3)
node5 = get_row(node1, 5)