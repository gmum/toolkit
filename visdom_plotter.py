#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Plots plots from experiment. Logical structure: set of handlers

Handlers:

    1. CSV model train: *.csv (plots all columns but first groups them by "_")
    2. Any image: *.png (if has index then just replaces, if not sents new)
    3. Config file: config.json gets dumped as text

Usage:

    ./visdom_plotter.py --folder=PATH_TO_FOLDER --env=NAME_OF_EXPERIMENT

Or in training script:

        ret = subprocess.Popen([os.path.join(os.path.dirname(__file__), "../visdom_plotter.py"),
            "--visdom-server={}".format(os.environ['VISDOM_SERVER']), "--folder={}".format(save_path)])
        time.sleep(0.1)
        if ret.returncode is not None:
            raise Exception()
        atexit.register(lambda: os.kill(ret.pid, signal.SIGINT))
"""

import argh
import time
import glob
import os
import json
import visdom

import matplotlib.pylab as plt

from scipy import misc # For image loading
import pandas as pd
import numpy as np
from collections import defaultdict
import traceback

N_TRIES = 10 # Tries 10 times before accepting failure

SENT_IMAGES = set()

def _succeed(visdom_out, win_name):
    # Visdom doesn't hove correct error handling ATM. Not sure why, is it too hard
    # to code error handling for FAIR researchers? :P
    return visdom_out == win_name

def _img_plots_from_folder(folder_name, vis):

    global SENT_IMAGES

    imgs = glob.glob(os.path.join(folder_name, "*png"))
    # TODO: Remove this hack
    imgs = [i for i in imgs if i not in SENT_IMAGES]
    SENT_IMAGES |= set(imgs)

    # print("Found {} images".format(len(imgs)))

    # Collect line plots
    data = defaultdict(list) # name -> [line_1, line_2, line_3..]
    max_id = {} # name -> id, we can have plots for each epoch

    for img_fname in imgs:
        img_name = os.path.splitext(os.path.basename(img_fname))[0]

        if len(img_name.split("_")) > 1:
            id = img_name.split("_")[-1]
            try:
                id = int(id)
                img_name = "_".join(img_name.split("_")[0:-1])
            except:
                img_name = img_name
        else:
            id = 0
            img_name = img_name

        assert id >= 0

        if id > max_id.get(img_name, -1):
            data[img_name] = misc.imread(img_fname)  # (2, 0, 1) for visdom convention
            max_id[img_name] = max(max_id.get(img_name, 0), id)

    for img_name in data:
    #     print data[img_name].shape
    #     print data[img_name].dtype
        data[img_name] = misc.imresize(data[img_name], (400, 400)).transpose(2, 0, 1)
    #     print data[img_name].shape

    # Plot them
    for plot_name in data:
        # print("Plotting " + plot_name)

        succeed = False
        for i in range(N_TRIES):
            out = vis.image(data[plot_name], win=plot_name, opts={"title": plot_name, "size": (400, 400)})
            succeed = _succeed(out, plot_name)
            if succeed:
                break
            else:
                time.sleep(1)

        if not succeed:
            print("Failed to plot " +plot_name)

def _line_imgs_from_csv(f_name, vis):
    try:
        H = pd.read_csv(f_name)
    except:
        print("Warning - failed reading csv")
        return

    bf_name = os.path.basename(f_name)

    # print("Columns {}".format(H.columns))

    # Collect line plots
    data = defaultdict(list) # name -> [line_1, line_2, line_3..]
    legend = defaultdict(list) # name -> legend

    # TODO(kudkudak): How to code following grouping in a clearer way?
    # Always interprets _ as sublines
    for c in H.columns:

        if c.startswith("Unnamed"): # A hack to omit anonimously saved columns
            continue
        # Default name
        name = c

        # Handle the val/train/test prefix (only for simple names)
        if len(c.split("_")) >= 2:
            if c.split("_")[0] in {"val", "valid", "dev", "test", "train"}:
                name = "_".join(c.split("_")[1:])

        # Handle the suffix
        # if len(name.split("_")) > 1:
        #     id = name.split("_")[-1]
        #     name = "_".join(name.split("_")[0:-1])

        data[name].append(np.array(H[c]))
        legend[name].append(c)

    # Grouping


    # print(data)

    # Plot them
    for plot_name in data:

        # print("Plotting " + plot_name)
        X = np.array([np.arange(len(data[plot_name][0])) for _ in data[plot_name]]).T
        Y = np.array(data[plot_name]).T
        # cmap = plt.get_cmap("coolwarm", len(data[plot_name]))
        # colors = np.array([cmap(i/float(len(data[plot_name]))) for i in range(len(data[plot_name]))])
        # colors *= 255
        # colors = colors.astype("int")
        # assert colors.shape[1] == 4
        # colors = colors[:, 0:3] # Remove alpha channel
        # print(Y)
        try:
            if Y.ndim == 2 and np.prod(Y.shape) > 1:
                succeed = False
                for i in range(N_TRIES):
                    out = vis.line(X=X, Y=Y, win=bf_name + plot_name,
                        opts={"title": bf_name + " " + plot_name,  "legend": legend[plot_name]})
                    succeed = _succeed(out, bf_name  + plot_name)
                    if succeed:
                        break
                    else:
                        print out, bf_name + " " + plot_name

                if not succeed:
                    pass
            else:
                pass
        except Exception, e:
            pass
            # traceback.print_exc()


def loop(folder=".", visdom_server="http://visdom.capdnet.ii.uj.edu.pl", visdom_port=80,
    png_handler=1):

    folder = folder.split(",")

    print("Streaming from " + str(folder))

    vis = {}

    for f in folder:
        env = f
        vis[f] = visdom.Visdom(visdom_server, port=visdom_port, env=env)

    try:
        while True:
            for f in folder:
                # Visualizer 1: line plots
                for csv_file in glob.glob(os.path.join(f, "*.csv")):
                    _line_imgs_from_csv(csv_file, vis[f])

                # Visualizer 2: img plots
                if png_handler:
                    _img_plots_from_folder(f, vis[f])

                # Visualizer 3: config plot
                if os.path.exists(os.path.join(f, "config.json")):
                    vis[f].text(json.dumps(json.load(open(os.path.join(f, "config.json"))),
                        indent=4, sort_keys=True), win="config", opts={"title": "config"})

                time.sleep(30)

                vis[f].save(envs=[f])
    except KeyboardInterrupt:
        print("Closing")

        # vis.close(env=env)

if __name__ == "__main__":
    argh.dispatch_command(loop)
