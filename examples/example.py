
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
def get_row(X, r):
    return X[r]

node1 = get_X()
node2 = get_row(node1, 3)