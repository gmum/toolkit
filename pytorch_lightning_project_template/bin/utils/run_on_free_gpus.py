#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple self-contained script to run on a list of jobs on all free GPUs on a machine

See bin/utils/run_on_free_gpus.py -h

There are the following requirements for the batch:
* Each command has saving dir as 2nd argument
* Each script saves to the save_dir HEARTBEAT
* Each script saves to the save_dir FINISHED when done
"""

print("Remember to multiple your number of jobs by factor of threads per cpu")

import os
import time
import numpy as np
from os import path
import pandas as pd
import argh
import subprocess
import logging

from src.utils import configure_logger
configure_logger('', log_file=None)
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))

logger = logging.getLogger(__name__)


def get_next_free_gpu():
    for i in range(10):
        try:
            output = subprocess.check_output(['nvidia-smi', '-i', str(i)]).decode("utf-8")
        except:
            logger.warning("Failed nvidia-smi {}".format(i))
            output = ""

        if output == "No devices were found":
            return None
        elif "No running processes found" in output:
            return i
        else:
            continue

import stat

def get_last_modification(save_path):
    f_path = os.path.join(save_path, "HEARTBEAT")
    stderr_path = os.path.join(save_path, "stderr.txt")
    if os.path.exists(f_path):
        return time.time() - os.stat(f_path)[stat.ST_MTIME]
    else:
        if os.path.exists(stderr_path):
            return time.time() - os.stat(stderr_path)[stat.ST_MTIME]
        else:
            return 10000000000000


def get_n_jobs(script_name="train"):
    try:
        output = subprocess.check_output('ps -f | grep {}'.format(script_name), shell=True).decode("utf-8").strip()
    except:
        print("No jobs")
        output = ""
    # print("===")
    # print(output)
    # print("===")
    return len(output.split("\n")) - 2

def get_save_path(job):
    assert "$" not in job, "Seems like there in env variable in the command"
    if job.startswith("python"):
        return job.split(" ")[2]
    else:
        return job.split(" ")[1]


def get_script_name(job):
    if job.startswith("python"):
        return job.split(" ")[1]
    else:
        return job.split(" ")[0]


def has_finished(save_path):
    # Hacky but works, usually
    return path.exists(path.join(save_path, "FINISHED"))


def get_jobs(batch):
    jobs = list(open(batch, "r").read().splitlines())
    jobs = [j for j in jobs if not has_finished(get_save_path(j))]
    # take only at least 10min old jobs
    jobs = [j for j in jobs if get_last_modification(get_save_path(j)) > 600]
    jobs = [("python " + j) if "python" not in j else j for j in jobs]
    np.random.shuffle(jobs)
    return jobs

def tensorboard_running():
    output = subprocess.check_output('ps | grep tensorboard', shell=True).decode("utf-8").strip()
    return len(output.split("\n")) > 1

def run(batch, max_jobs=1):
    try:
        if len(get_jobs(batch)) == 0:
            logger.error("No untouched (>10min old) jobs found. Exiting.")
            exit(1)

        tb_dir = os.path.join(RESULTS_DIR, "running_experiments")
        os.system("mkdir -p " +tb_dir)

        save_path = get_save_path(get_jobs(batch)[0])
        script_name = get_script_name(get_jobs(batch)[0])

        root_save_path = path.dirname(save_path)
        os.system("rm " + root_save_path + " " + tb_dir)
        os.system("ln -s " + root_save_path + " " + tb_dir)
        logger.info("Running tensorboard in " + tb_dir)
        os.system("tensorboard --port=7777 --logdir=" + tb_dir+ " &")

        while True:
            print("next_free_gpu={},n_jobs running={},batch={}".format(get_next_free_gpu(), get_n_jobs(script_name), batch))
            jobs = get_jobs(batch)
            logger.info("Found {}".format(len(jobs)))
            if len(jobs):
                job = jobs[0]
                gpu = get_next_free_gpu()
                n_jobs = get_n_jobs(script_name)
                if gpu is not None and max_jobs > n_jobs:
                    os.system("mkdir -p " + get_save_path(job))
                    # Run and redirect all output to a file in the save folder of the job
                    logger.info("Running " + job)
                    os.system("CUDA_VISIBLE_DEVICES={} {}".format(gpu, job) + "> {} 2>&1".format(os.path.join(get_save_path(job), "last_run.out")) + " &")
                    while get_last_modification(get_save_path(job)) > 600 or get_next_free_gpu() == gpu:
                        print("Waiting for bootup (no HEARTBEAT or not occupied GPU)... last_mod={},next_free_gpu={},n_jobs={}".format(
                            get_last_modification(get_save_path(job)), get_next_free_gpu(), get_n_jobs(script_name)))
                        time.sleep(1)
                elif gpu is None:
                    logger.warning("No free GPUs")
                elif max_jobs <= n_jobs:
                    logger.warning("Have {} jobs  running but can run max max_jobs={}".format(get_n_jobs(script_name), max_jobs))
                else:
                    raise NotImplementedError()
            else:
                logger.info("No jobs found")
                if  get_n_jobs(script_name)==0:
                    exit(0)
            # Allow to kill easilyq
            time.sleep(5)
    except KeyboardInterrupt:
        logger.warning("Interrupt. Killing all python & tensorboard jobs.")
        os.system("ps | grep python |awk '{print $1}' | xargs kill -9")
        os.system("ps | grep tensorboard |awk '{print $1}' | xargs kill -9")
        os.system("rm " + os.path.join(tb_dir, path.basename(root_save_path)))

if __name__ == "__main__":
    argh.dispatch_command(run)
