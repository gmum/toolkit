#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple script to run on a list of jobs on a GPU

See bin/utils/run_on_a_gpu.py -h

There are the following requirements for the batch:
* Each command has saving dir as 2nd argument
* Each script saves to the save_dir HEARTBEAT
* Each script saves to the save_dir FINISHED when done
"""

import os
import time
import numpy as np
from os import path
import pandas as pd
import argh

from src import configure_logger  # Will actually configure already

import logging

logger = logging.getLogger(__name__)


def get_last_modification(save_path):
    f_path = os.path.join(save_path, "HEARTBEAT")
    if os.path.exists(f_path):
        return time.time() - os.path.getmtime(f_path)
    else:
        return np.inf


def get_save_path(job):
    if job.startswith("python"):
        return job.split(" ")[3]
    else:
        return job.split(" ")[2]


def has_finished(save_path):
    # Hacky but works, usually
    return path.exists(path.join(save_path, "FINISHED"))


def get_jobs(batch):
    jobs = list(open(batch, "r").read().splitlines())
    jobs = [j for j in jobs if not has_finished(get_save_path(j))]
    # take only at least 10min old jobs
    jobs = [j for j in jobs if get_last_modification(get_save_path(j)) > 600]
    np.random.shuffle(jobs)
    return jobs

def shell_single(batch, gpu=-1):
    # Assumes gpu is configured
    while True:
        jobs = get_jobs(batch)
        logger.info("Found {}".format(len(jobs)))
        job = jobs[0]
        logger.info("Running " + job)
        if gpu==-1:
            os.system(job)
        else:
            os.system("CUDA_VISIBLE_DEVICES={} {}".format(gpu, job))
        # Allow to kill easilyq
        time.sleep(5)


if __name__ == "__main__":
    argh.dispatch_command(shell_single)
