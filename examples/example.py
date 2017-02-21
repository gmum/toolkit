


import numpy as np

@
def get_X():
    return np.arange(50).reshape(10,5)


def get_row(X, r):
    return X[r]

node1 = get_X()
node2 = get_row(node1, 3)