# Utils for plotting.
import json
import logging
import os
import pickle

import matplotlib as mpl
import matplotlib.style
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import copy
import gin
import time

from gin.config import _CONFIG

from src.utils import configure_neptune_exp, get_neptune_exp, configure_logger
from src import PROJECT_NAME
from os.path import join

logger = logging.getLogger(__name__)


def load_C(E):
    return load_HC(E)[1]


def load_HbC(e):
    # A utility function to load H (per batch) and C
    if len(_CONFIG):
        logger.warning("Erasing global gin config")
        gin.clear_config()
    H = None
    for i in range(3):
        try:
            H = pickle.load(open(join(e, "history_batch.pkl"), "rb"))
            H_epoch = pickle.load(open(join(e, "history.pkl"), "rb"))
        except:
            print("Warning. Faied to read :" + e)
            time.sleep(1)
            continue
    assert H is not None, "Failed to read " + e
    for k in H_epoch:
        H['epoch_' + k] = H_epoch[k]
    for k in H:
        H[k] = np.array(H[k])
    gin.parse_config(open(join(e, "config.gin")))
    C = copy.deepcopy(_CONFIG)
    C = {k[1].split(".")[-1]: v for k, v in C.items()}  # Hacky way to simplify config
    return H, C


def load_HC(e, force_pkl=False):
    # A utility function to load H and C
    if len(_CONFIG):
        logger.warning("Erasing global gin config")
        gin.clear_config()
    H = None
    for i in range(3):
        try:
            if os.path.exists(join(e, 'history.csv')) and not force_pkl:
                H = pd.read_csv(join(e, "history.csv"))
            else:
                H = pickle.load(open(join(e, "history.pkl"), "rb"))
        except:
            print("Warning. Faied to read :" + e)
            time.sleep(1)
            continue
    assert H is not None, "Failed to read" + e
    gin.parse_config(open(join(e, "config.gin")))
    C = copy.deepcopy(_CONFIG)
    C = {k[1].split(".")[-1]: v for k, v in C.items()}  # Hacky way to simplify config
    for k in list(_CONFIG):
        del _CONFIG[k]
    return H, C


def construct_colorer(sorted_vals, cmap="coolwarm"):
    cm = plt.get_cmap(cmap, len(sorted_vals))
    N = float(len(sorted_vals))

    def _get_cm(val):
        return cm(sorted_vals.index(val) / N)

    return _get_cm


def construct_marker(sorted_vals):
    cm = ['o', 'x', 'd']
    N = float(len(sorted_vals))

    def _get_cm(val):
        return cm[sorted_vals.index(val)]

    return _get_cm


def construct_colorer_lin_scale(vmin, vmax, ticks=20, cmap="coolwarm"):
    assert vmax > vmin

    cm = plt.get_cmap(cmap, ticks)

    def _get_cm(val):
        alpha = (val - vmin) / float((vmax - vmin))
        tick = int(alpha * ticks)
        tick = min(tick, ticks - 1)
        return cm(tick)

    return _get_cm


try:
    import dropbox
    access_token = os.environ['DROPBOXTOKEN']

    class TransferData:
        def __init__(self, access_token):
            self.access_token = access_token

        def upload_file(self, file_from, file_to):
            """upload a file to Dropbox using API v2
            """
            dbx = dropbox.Dropbox(self.access_token)
            with open(file_from, 'rb') as f:
                try:
                    dbx.files_upload(f.read(), file_to, mode=dropbox.files.WriteMode.overwrite)
                except:
                    logger.error("Failed uploading " + file_from + " " + file_to)


    transferData = TransferData(access_token)
except:
    transferData = None


def save_fig(dir_name, figure_name, copy_to_dropbox=False, copy_to_neptune=False):
    if figure_name.endswith("pdf"):
        figure_name = figure_name[0:-4]
    figure_name = figure_name.replace("=", "").replace(":", "").replace("_", "").replace(".", "") \
        .replace("/", "_").replace("$", "").replace("\\", "")
    path = os.path.join(dir_name, figure_name + ".pdf")

    if not os.path.exists(os.path.dirname(path)):
        os.system("mkdir -p " + os.path.dirname(path))

    logger.info('Figure saved to: ' + path)

    fig = plt.gcf()
    fig.savefig(path, bbox_inches='tight',
                transparent=True,
                pad_inches=0)

    if copy_to_dropbox:
        # / because it is a dropbox app with own folder
        transferData.upload_file(path, os.path.join("/", PROJECT_NAME, path))

    if copy_to_neptune:
        neptune_exp = get_neptune_exp()
        neptune_exp.send_image(figure_name, fig)


if __name__ == "__main__":
    configure_logger('')
    configure_neptune_exp('tst')
    plt.plot([1,2,3], [1,2,4])
    save_fig("examples", "qudratic.pdf", copy_to_dropbox=True, copy_to_neptune=True)
    plt.show()
    plt.close()