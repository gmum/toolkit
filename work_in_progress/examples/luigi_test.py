import sys
import numpy as np

from burrito import wrap

def get_X(n=0):
    A = np.arange(n).reshape(10,5)
    return "OK"

import luigi

class WordCount(luigi.Task):
    date_interval = luigi.DateIntervalParameter()
    


if __name__ == "__main__":
   wrap(get_X)
