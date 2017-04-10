#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Plots plots from experiment. Logical structure: set of handlers

Handlers:

    1. *.csv (plots all columns but first groups them by "_" if characters after "_" are forming int)
    2. *.png (if has index then just replaces, if not sents new)
    3. config.json gets dumped as text

Usage:

    ./visdom_plotter.py --folder=PATH_TO_FOLDER --env=NAME_OF_EXPERIMENT

TODO:
    1. Pull default visdom_server from env variables
    2. More handlers
"""

import argh
import time
import glob
import os
import json
import visdom

from scipy import misc # For image loading
import pandas as pd
import numpy as np
from collections import defaultdict

N_TRIES = 10 # Tries 10 times before accepting failure

def _succeed(visdom_out, win_name):
    # Visdom doesn't hove correct error handling ATM. Not sure why, is it too hard
    # to code error handling for FAIR researchers? :P
    return visdom_out == win_name

def _img_plots_from_folder(folder_name, vis):
    imgs = glob.glob(os.path.join(folder_name, "*png"))

    print("Found {} images".format(len(imgs)))

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
        print data[img_name].shape
        print data[img_name].dtype
        data[img_name] = misc.imresize(data[img_name], (200, 200)).transpose(2, 0, 1)
        print data[img_name].shape

    # Plot them
    for plot_name in data:
        print("Plotting " + plot_name)

        succeed = False
        for i in range(N_TRIES):
            out = vis.image(data[plot_name], win=plot_name, opts={"title": plot_name, "size": (200,200)})
            succeed = _succeed(out, plot_name)
            if succeed:
                break

        if not succeed:
            print("Failed to plot " +plot_name)

def _line_imgs_from_csv(f_name, vis):
    H = pd.read_csv(f_name)

    print("Columns {}".format(H.columns))

    # Collect line plots
    data = defaultdict(list) # name -> [line_1, line_2, line_3..]
    legend = defaultdict(list) # name -> legend

    for c in H.columns: # TODO: Fix [1:]

        if len(c.split("_")) > 1:
            id = c.split("_")[-1]
            try:
                id = int(id)
                name = "_".join(c.split("_")[0:-1])
            except:
                name = c
        else:
            name = c

        data[name].append(np.array(H[c]))
        legend[name].append(c)

    print(data)

    # Plot them
    for plot_name in data:
        print("Plotting " + plot_name)
        X = np.array([np.arange(len(data[plot_name][0])) for _ in data[plot_name]]).T
        Y = np.array(data[plot_name]).T

        succeed = False
        for i in range(N_TRIES):
            out = vis.line(X=X, Y=Y, win=plot_name, opts={"title": plot_name, "legend": legend[plot_name]})
            print(out)
            succeed = _succeed(out, plot_name)
            if succeed:
                break

        if not succeed:
            print("Failed to plot " +plot_name)


def loop(folder=".", env="main", visdom_server="http://visdom.capdnet.ii.uj.edu.pl", visdom_port=80):
    vis = visdom.Visdom(visdom_server, port=visdom_port, env=env)

    try:
        while True:
            # Visualizer 1: line plots
            for csv_file in glob.glob(os.path.join(folder, "*.csv")):
                _line_imgs_from_csv(csv_file, vis)

            # Visualizer 2: img plots
            _img_plots_from_folder(folder, vis)

            # Visualizer 3: config plot
            if os.path.exists(os.path.join(folder, "config.json")):
                vis.text(json.dumps(json.load(open(os.path.join(folder, "config.json"))),
                    indent=4, sort_keys=True), win="config", opts={"title": "config"})

            time.sleep(5)
    except:
        print("Closing")
        vis.close(env=env)

if __name__ == "__main__":
    argh.dispatch_command(loop)
