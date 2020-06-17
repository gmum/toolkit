#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple self-contained script template (you need to update paths) to update and remove unused figures in a tex file
"""
import glob
import os
import tqdm

# Configure this
PATH_TO_TEX = "papers/entanglement/main.tex"
SRC1 = "experiments/04_2020_sweeps"
SRC2 = "experiments/05_2020_understand_grads"
SRCS = [SRC1, SRC2]
REMOVE_UNUSED_FIGURES = True
PAPER_SRCS = ["papers/entanglement/figs/*pdf"]

# Go through all sources and update if there is a newer pdf
TEX = open(PATH_TO_TEX).read()
for SRC in SRCS:
    for SOURCE in PAPER_SRCS:
        files = list(glob.glob(SOURCE))
        for f in tqdm.tqdm(files, total=len(files)):
            if os.path.basename(f) not in TEX:
                print("WARNING! {} not found in main.tex".format(f))
                if REMOVE_UNUSED_FIGURES:
                    os.system("rm " + f)
            else:
                print("OK")
            a = os.path.join(SRC, os.path.basename(f))

            if "@" in SRC1:
                os.system("scp {} {}".format(a, f))
            else:
                os.system("cp {} {}".format(a, f))
