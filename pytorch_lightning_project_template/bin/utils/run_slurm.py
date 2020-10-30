#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple self-contained script to run on a list of jobs on a SLURM system

You will need to update bin/utils/slurm_template.sh

See bin/utils/run_slurm.py -h

There are the following requirements for the batch:
* Each command has saving dir as 2nd argument
* Each script saves to the save_dir HEARTBEAT
* Each script saves to the save_dir FINISHED when done

When using: figure out which queue is better. Bsub doesn't like multiple queues.

When adapting to a new project make sure that get_save_path is OK, and that you replace the project name.
"""

import os
import time
import stat
from os import path
import subprocess
import logging
import random
import string
from logging import handlers
import sys

def configure_logger(name='',
                     console_logging_level=logging.INFO,
                     file_logging_level=None,
                     log_file=None):
    """
    Configures logger
    :param name: logger name (default=module name, __name__)
    :param console_logging_level: level of logging to console (stdout), None = no logging
    :param file_logging_level: level of logging to log file, None = no logging
    :param log_file: path to log file (required if file_logging_level not None)
    :return instance of Logger class
    """

    if file_logging_level is None and log_file is not None:
        print("Didnt you want to pass file_logging_level?")

    if len(logging.getLogger(name).handlers) != 0:
        print("Already configured logger '{}'".format(name))
        return

    if console_logging_level is None and file_logging_level is None:
        return  # no logging

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if console_logging_level is not None:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(format)
        ch.setLevel(console_logging_level)
        logger.addHandler(ch)

    if file_logging_level is not None:
        if log_file is None:
            raise ValueError("If file logging enabled, log_file path is required")
        fh = handlers.RotatingFileHandler(log_file, maxBytes=(1048576 * 5), backupCount=7)
        fh.setFormatter(format)
        logger.addHandler(fh)

    logger.info("Logging configured!")

    return logger



configure_logger('', log_file=None)

RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))
TIMEOUT = 180 # 3 minutes. Might be short for some

logger = logging.getLogger(__name__)


def random_string(strlen=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(strlen))


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

def is_running(job_id):
    # A hacky way to chekc if a job is running
    output = subprocess.check_output("squeue -u jastrs01", shell=True).decode("utf-8").strip()
    if output.find(str(job_id)) == -1:
        return False
    else:
        return True

def get_n_jobs(batch_name):
    try:
        output = subprocess.check_output('squeue -u jastrs01 | grep {}'.format(batch_name), shell=True).decode("utf-8").strip()
    except:
        print("No jobs")
        output = ""
    if len(output) == 0:
        return 0
    else:
        return len(output.split("\n")) - 1 # -1 Because header

def get_save_path(job):
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
    if ";" in batch:
        batches = batch.split(";")
    else:
        batches = [batch]

    all_jobs = []
    for batch in batches:
        jobs = list(open(batch, "r").read().splitlines())
        jobs = [j for j in jobs if j[0] != "#"]
        jobs = [j for j in jobs if not has_finished(get_save_path(j))]
        # take only at least 10min old jobs
        jobs = [j for j in jobs if get_last_modification(get_save_path(j)) > 600]
        jobs = [("python " + j) if "python" not in j else j for j in jobs]
        all_jobs += jobs

    random.shuffle(all_jobs)
    return all_jobs

def tensorboard_running():
    output = subprocess.check_output('ps | grep tensorboard', shell=True).decode("utf-8").strip()
    return len(output.split("\n")) > 1


def run_job(job, batch_name, exclude_hosts=[], wait=1):
    # Shorter jobs !
    slurm_cmd=open("bin/utils/slurm_template.sh", "r").read()
    slurm_cmd=slurm_cmd.format(job=job, batch_name=batch_name, save_path=get_save_path(job))
    with open(os.path.join(get_save_path(job), "run.sh"), "w") as f:
        logger.info("Writing runner to " + os.path.join(get_save_path(job), "run.sh"))
        f.write(slurm_cmd)
    logger.info("Submitting job to bsub")

    # A heuristic way to submit the job
    cmd = "sbatch {}".format(os.path.join(get_save_path(job), "run.sh"))
    output = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    assert output.startswith("Submitted")
    assert len(output.split(" ")) == 4
    job_id = int(output.split(" ")[-1])
    logger.info("Job id is " + str(job_id))

    # A hacky way to get hostname
    hostname = ""
    while wait:
        logger.info("Waiting to find a machine for {}.. last hostname is {}".format(cmd, hostname))

        output = subprocess.check_output("squeue -u jastrs01 | grep " + str(job_id), shell=True).decode("utf-8").strip()
        parsed_output = output.strip().split()
        assert parsed_output[0] == str(job_id)
        if parsed_output[-1][0] != "(":
            hostname = parsed_output[-1]
            break

        time.sleep(1)

    if hostname != "":
        logger.info("Hostname is " + str(hostname))

    return job_id, hostname


def run(batch, max_jobs=1, wait=1):
    exclude_hosts = []

    batch_name = random_string(5)

    n_jobs_start = len(get_jobs(batch))

    print("Starting")
    print("==")

    try:
        if len(get_jobs(batch)) == 0:
            logger.error("No untouched (>10min old) jobs found. Exiting.")
            exit(1)

        tb_dir = os.path.join(RESULTS_DIR, "running_experiments")
        os.system("mkdir -p " +tb_dir)

        save_path = get_save_path(get_jobs(batch)[0])
        root_save_path = path.dirname(save_path)
        os.system("rm " + root_save_path + " " + tb_dir)
        os.system("ln -s " + root_save_path + " " + tb_dir)

        while True:
            print("n_jobs={}/{},batch={},\nexclude_hosts={},name={}".format(get_n_jobs(batch_name), n_jobs_start, batch, exclude_hosts,batch_name))
            jobs = get_jobs(batch)
            logger.info("Found {} jobs to run in the batch script.".format(len(jobs)))
            if len(jobs):
                job = jobs[0]
                n_jobs = get_n_jobs(batch_name)
                if max_jobs > n_jobs:
                    os.system("mkdir -p " + get_save_path(job))
                    # Run and redirect all output to a file in the save folder of the job
                    logger.info("Running " + job)
                    job_id, hostname = run_job(job, batch_name, exclude_hosts, wait)
                    start_wait = time.time()
                    while get_last_modification(get_save_path(job)) > 600 and wait:
                        print("Waiting for bootup (no HEARTBEAT)... last_mod={},n_jobs={}".format(
                            get_last_modification(get_save_path(job)), get_n_jobs(batch_name)))

                        if not is_running(job_id):
                            print("Job died. Probably a faulty machine or a bug in code. Rerunning")
                            print("Job cmd was: " + job)
                            break

                        time.sleep(1)

                        if time.time() - start_wait > TIMEOUT:
                            logger.info("Couldn't start the job in {}s. Killing and maybe excluding host (WARNING: excluding doesnt work for slurm).".format(TIMEOUT))
                            os.system("scancel " + str(job_id))
                            break
                elif max_jobs <= n_jobs:
                    logger.warning("Have {} jobs  running but can run max max_jobs={}".format(get_n_jobs(batch_name), max_jobs))
                else:
                    raise NotImplementedError()
            else:
                logger.info("No jobs found")
            # Allow to kill easilyq
            time.sleep(5)
    except KeyboardInterrupt:
        os.system("scancel -n " + batch_name)

if __name__ == "__main__":
    _, batch, n_jobs = sys.argv
    run(batch, int(n_jobs))
