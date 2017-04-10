#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 Standalone module slightly extending multiprocessing.Pool with
 caching, easy interruption and logging

 Includes two extensions of map and JobPool which is basically improved map + json bookkeeping
"""

import json
import logging
import os
import time
import sys
# Pebble supports querying individual jobs and passing kwargs
from pebble.process import Pool
from pebble import TimeoutError
from os import path
import signal
import tqdm
import glob

logger = logging.getLogger(__name__)


def dict_hash(my_dict):
	return str(abs(hash(frozenset(my_dict.items()))))



def smart_map(fnc, jobs, n_jobs=10, job_timeout=None, flush_to_stdout=False):
	# Run (sync) list of jobs without worrying of killing your server or notebook
	pool = Pool(n_jobs)
	try:
		with tqdm.tqdm(total=len(jobs)) as bar:
			scheduled_jobs = []
			for j in jobs:
				if isinstance(j, list) or isinstance(j, tuple):
					scheduled_jobs.append(pool.schedule(fnc, args=j, timeout=job_timeout))
				elif isinstance(j, dict):
					scheduled_jobs.append(pool.schedule(fnc, kwargs=j, timeout=job_timeout))
				else:
					scheduled_jobs.append(pool.schedule(fnc, args=[j], timeout=job_timeout))

			while True:
				remaining = sum(not j.ready for j in scheduled_jobs)
				if remaining == 0:
					logger.info("Done")
					break

				if (len(jobs) - remaining) - bar.n:
					bar.update((len(jobs) - remaining) - bar.n)
					avg_time = bar.avg_time
					if flush_to_stdout and avg_time is not None:
						logger.info("Remaining {} jobs, est. time {:.2f}s".format(remaining, avg_time * remaining))
				time.sleep(3)
		res = [j.get() for j in scheduled_jobs]
		pool.close(); pool.join()
		return res
	except KeyboardInterrupt, e:
		logger.warning("Terminating!")
		pool.stop()
		return None


def smart_map_with_cache(fnc, jobs, keys, cache, n_jobs=1, job_timeout=None, flush=True):
	"""
	Parameters
	----------
	fnc: function

	jobs: list
		Each element of list is parsed differently depending on its type. Dict is assumed to be kwargs,
		list and tuple are assumed to be positional args and any other type is interpreted as single argument

	keys: list

	cache: str or SmartDataset or dict
		If str will write to cache/job_key anything outputted by function (will in the future use extension
		to infer format, for now please do conversion on your own)

		If SmartDataset will save to it every 600s

		If dict will save to it, but won't persist it

	n_jobs:

	job_timeout: int, defaut:None
		Terminates jobs longer than job_timeout, won't write them to cache

	flush: bool, default:False
		If true will flush to stdout time stimations

    Returns
	-------
		output: dict
			Calculated k -> returned_value dict
	"""
	logging.info("Loading cache")

	# TODO(kudkudak): Worth reimplementing as separate class (with load/save fncs)
	if hasattr(cache, "load") and hasattr(cache, "save"):
		# TODO(kudkudak): After refactoring into sep. class remove this try/catch
		# it is essentioanlly hack to support SmartDataset without saved state
		try:
			D = cache.load()
		except Exception, e:
			logging.warning("Failed loading cache with exception {}".format(e))
			D = {}
	elif isinstance(cache, str):
		if not os.path.isdir(cache):
			try:
				os.makedirs(cache)
			except:
				raise RuntimeError("Failed to create directory with cache on " + cache)

		failed, D = 0, {}
		for path in glob.glob(os.path.join(cache, "*")):
			try:
				key = os.path.basename(path)
				if key in keys:
					# TODO(kudkudak): Add file format support
					D[key] = open(path).read()
			except:
				failed += 1
		if failed > 0:
			logging.warning("Failed loading {} jsons".format(failed))
		logging.info("Loaded {} jsons from {}".format(len(D), cache))
	elif isinstance(cache, dict):
		logging.warning("Partial saving is not supported for dict. This means all of the computation is thrown away"
						"if any jobs raises error.")
		D = cache
	else:
		raise NotImplementedError("Not recognized cache type")

	# Filter out calculated jobs
	logging.info("Skipping {}/{} calculated jobs".format(sum(k in D for k in keys), len(keys)))
	jobs, keys = [j for k, j in zip(keys, jobs) if k not in D], [k for k in keys if k not in D]

	# Run everything
	pool = Pool(n_jobs)
	time_elapsed = 0
	try:
		with tqdm.tqdm(total=len(jobs)) as bar:
			scheduled_jobs = []
			# Schedule jobs
			for j in jobs:
				if isinstance(j, list) or isinstance(j, tuple):
					scheduled_jobs.append(pool.schedule(fnc, args=j, timeout=job_timeout))
				elif isinstance(j, dict):
					scheduled_jobs.append(pool.schedule(fnc, kwargs=j, timeout=job_timeout))
				else:
					scheduled_jobs.append(pool.schedule(fnc, args=[j], timeout=job_timeout))

			# Run calculation
			while True:

				remaining = sum(not j.ready for j in scheduled_jobs)
				if remaining == 0:
					logger.info("Done")
					break

				if (len(jobs) - remaining) - bar.n:
					bar.update((len(jobs) - remaining) - bar.n)
					if flush and bar.avg_time is not None:
						logging.info("Remaining {} jobs, est. time {:.2f}s".format(remaining, bar.avg_time * remaining))
						sys.stderr.flush()

				time.sleep(3)
				time_elapsed += 3
				# Save to cache
				# TODO(kudkudak): Worth reimplementing as separate class
				for k, s_j, j in zip(keys, scheduled_jobs, jobs):
					if (s_j.ready) and (k not in D):

						# If job failed it raise its error on s_j.get()
						# else gather it result
						try:
							D[k] = s_j.get()
						except TimeoutError, e:
							logging.warning("Timed out {}".format(k))
						except Exception, e:
							logger.error("Failed running job with key '{}' and params {}".format(k, j))
							raise e

						# And add incrementally to cache
						if k in D and D[k] is not None:
							if isinstance(cache, str):
								open(os.path.join(cache, k), "w").write(D[k])
							elif hasattr(cache, "load") and hasattr(cache, "save"):
								pass
							elif isinstance(cache, dict):
								pass
							else:
								raise NotImplementedError("Not recognized cache type")

				if hasattr(cache, "load") and hasattr(cache, "save"):
					# Save every minute to cache
					if time_elapsed % 60 == 0:
						logging.info("Dumping to cache, time_elapsed=" + str(time_elapsed))
						sys.stderr.flush()
						# Disabling KeyboardInterrupt for saving cache
						s = signal.signal(signal.SIGINT, signal.SIG_IGN)
						cache.save(D)
						signal.signal(signal.SIGINT, s)

		if hasattr(cache, "load") and hasattr(cache, "save"):
			s = signal.signal(signal.SIGINT, signal.SIG_IGN)
			cache.save(D)
			signal.signal(signal.SIGINT, s)

		return D

	except KeyboardInterrupt:
		logger.warning("Terminating!")
		pool.stop()
		return D


class JobPool(object):
	"""
	Wrapper over map_with_cache, which does more json bookkeeping and makes sure nothing
	is overwritten or there are unknown jobs in a directory

	TODO(kudkudak): Add json book-keeping file inside output foldr
	TODO(kudkudak): Add support for multimachine (or maybe IPython parallel is better for that?)
	"""

	def __init__(self,
				 output_dir,
				 n_jobs=1,
				 additional_files_policy="delete",
				 conflict_policy="skip",
				 flush_to_stdout=False):
		if not conflict_policy in ['skip', 'raise']:
			raise RuntimeError("Unsupported conflict_policy")

		if not additional_files_policy in ['skip', 'delete']:
			raise RuntimeError("Unsupported additional_files_policy")

		self.output_dir = output_dir
		self.flush_to_stdout = flush_to_stdout
		self.conflict_policy = conflict_policy
		self.n_jobs = n_jobs
		self.additional_files_policy = additional_files_policy
		self.load_output_fnc = lambda output_fname: json.loads(open(output_fname).read())

	def map(self, fnc, jobs, keys):
		"""
		Params
		------
		conflict_policy: str
			in 'skip' or 'recalculate'

		Notes
		-----
		Loads jsons after running function
		"""
		if not all(isinstance(j, dict) for j in jobs):
			raise RuntimeError("Incorrect type of job, JobPool supports only dicts")

		if not os.path.exists(self.output_dir):
			ret = os.system("mkdir -p " + self.output_dir)
			assert ret == 0, "Failed to create dir " + self.output_dir

		# 1. Basically removes failed + non-matching predefined list jsons
		# and logs how many already calculated
		jobs_to_run, keys_to_run, removed_failed, to_delete = [], [], 0, []
		all_outputs = set()
		all_jsons = set(glob.glob(os.path.join(self.output_dir, "*.json")))
		for job_id, (key, job) in enumerate(zip(keys, jobs)):
			job = dict(job)  # We will modify it
			job_id = len(jobs_to_run)
			output_fname = path.join(self.output_dir, key + ".json")
			job.update({"_job_id": job_id, "_job_key": key, "_job_output_fname": output_fname,
						"_job_output_dir": self.output_dir})
			all_outputs.add(output_fname)

			if os.path.exists(
					output_fname) and not self.conflict_policy == 'recalculate':
				if os.stat(output_fname).st_size == 0:
					jobs_to_run.append(job)  # Might happen rarely
					keys_to_run.append(key)
					removed_failed += 1
					assert os.system(
						"rm " + output_fname) == 0, "Successfuly erased file"
				else:
					# Try reading it
					try:
						self.load_output_fnc(output_fname)
					except Exception, e:
						logger.warning(
							"Failed loading {} with {}".format(output_fname, e))
						logger.warning("First line: {}".format(
							open(output_fname).readline()))
						to_delete.append(output_fname)
			else:
				if self.conflict_policy == 'recalculate':
					to_delete.append(output_fname)

				jobs_to_run.append(job)  # Might happen rarely
				keys_to_run.append(key)

		if not all_jsons <= all_outputs and self.additional_files_policy == 'delete':
			logger.error("Folder {} has jsons it probably shouldn't".format(self.output_dir))
			logger.error("Example " + list(all_jsons - all_outputs)[0])
			if raw_input(
					"Type Y if you want to remove {} jsons".format(
						len(all_jsons - all_outputs))) == "Y":
				for fname in list(all_jsons - all_outputs):
					assert os.system(
						"rm " + fname) == 0, "Failed removing non-empty json"

		if len(to_delete):
			if raw_input(
					"Type Y if you want to remove {} (non-empty) jsons".format(
						len(to_delete))) == "Y":
				for fname in to_delete:
					assert os.system(
						"rm " + fname) == 0, "Failed removing non-empty json"

		logging.info(jobs)
		logging.info(keys)
		if len(jobs) - len(jobs_to_run):
			logger.warning("Will skip calculation of " + str(
				len(jobs) - len(jobs_to_run)) + " jobs (already calculated)")

		try:
			results = smart_map_with_cache(fnc, jobs_to_run, keys_to_run, self.output_dir, \
										   n_jobs=self.n_jobs, flush=self.flush_to_stdout)
		except Exception, e:
			logger.error("Failed running at least one of the jobs")
			raise e

		# If fnc returns None it is reloaded from json file
		for k in results:
			if results[k] is None:
				assert os.path.exists(
					os.path.join(self.output_dir, k + ".json")), "Missed job, is runner saving correctly results?"
				results[k] = json.load(open(os.path.join(self.output_dir, k + ".json")))

		return results


def burrito_runner(script_name, script_params, **kwargs):
	# Assumes script is wrapped in src.script_wrapper
	script_params['base_fname'] = os.path.join(kwargs['_job_output_dir'], kwargs['_job_key'])

	cmd = "{} {}".format(script_name, " ".join(
		"--{} {}".format(k, v) for k, v in script_params.iteritems()))

	# Return stdout
	_, stderr, ret = exec_command(cmd, flush_stdout=False)

	if ret != 0:
		logging.error("\n".join(stderr))
		with open("last_failed.cmd", "w") as f:
			f.write(cmd)
		raise RuntimeError("Failed running cmd")


def shell_script_runner(script_name, script_params):
	cmd = "{} {}".format(script_name, " ".join(
		"--{} {}".format(k, v) for k, v in script_params.iteritems()))

	# Return stdout
	_, stderr, ret = exec_command(cmd, flush_stdout=False)

	if ret != 0:
		logging.error("\n".join(stderr))
		with open("last_failed.cmd", "w") as f:
			f.write(cmd)
		raise RuntimeError("Failed running cmd")


if __name__ == "__main__":
	# Simple demonstration, run after cd'ing into dr.gmum folder

	sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
	from drgmum.toolkit.utils import exec_command
	from drgmum import REPOSITORY_DIR

	RESULTS_DIR = os.path.join(REPOSITORY_DIR, "results")


	# # <><><><><><><> Run using smart_map <><><><><><><><><><><>#
    #
	# def number_add(a, b):
	# 	return a + b
    #
    #
	# print smart_map(number_add, [[1, b] for b in [1, 2, 3]])
    #
    #
	# # <><><><><><><> Run using smart_map_with_cache <><><><><><><><><><><>#
    #
	# def number_add(a, b):
	# 	return a + b
    #
    #
	# print smart_map_with_cache(number_add, [{"a": 1, "b": b} for b in [1, 2, 3]], [str(b) for b in [1, 2, 3]],
	# 						   "number_add_experiment_1")
	# print smart_map_with_cache(number_add, [[1, b] for b in [1, 2, 3]], [str(b) for b in [1, 2, 3]],
	# 						   "number_add_experiment_1")


	# <><><><><><><> Run using JobPool <><><><><><><><><><><><><>#

	def runner(script_name, script_params, **kwargs):
		# Assumes script is wrapped in src.script_wrapper
		script_params['base_fname'] = os.path.join(kwargs['_job_output_dir'], kwargs['_job_key'])

		cmd = "{} {}".format(script_name, " ".join(
			"--{} {}".format(k, v) for k, v in script_params.iteritems()))

		# Return stderr. Flushing it because it contains logs
		_, stderr, ret = exec_command(cmd, flush_stdout=False, flush_stderr=True)
		if ret != 0:
			raise RuntimeError("Failed running cmd")


	pool = JobPool(n_jobs=10, output_dir="number_add_experiment_2")
	jobs = [{"script_name": "examples/toolkit/number_add.py", "script_params": {"a": 1, "b": b}} for b in [1, 2, 3]]
	# JobPool handles json files itself, so keys is without ".json" extension
	keys = ["lol" + str(id) for id, j in enumerate(jobs)]
	results = pool.map(runner, jobs, keys)

	print results
