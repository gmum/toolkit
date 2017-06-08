# -*- coding: utf-8 -*-

"""
Random useful commands
"""

import sys, subprocess
from six import string_types
import threading
import json
import logging

_stdout_log = ""
_stderr_log = ""
# -*- coding: utf-8 -*-

"""
Random useful commands

TODO(kudkudak): Make this implementation more robust. Not working really well sometimes, especially for multirpcoess setting.
"""

import sys, subprocess
from six import string_types
import threading
import json
import logging

_stdout_log = ""
_stderr_log = ""

def stdout_thread(pipe, flush_stdout=False):
    global _stdout_log
    for line in pipe.stdout.readlines():
        _stdout_log += line + "\n"
        if flush_stdout:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()


def stderr_thread(pipe, flush_stderr=False):
    global _stderr_log
    for line in pipe.stderr.readlines():
        _stderr_log += line + "\n"
        if flush_stderr:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()

def exec_command(command, flush_stdout=False, flush_stderr=False, cwd=None, timeout=None):
    global _stdout_log, _stderr_log
    _stdout_log, _stderr_log = "", ""

    if isinstance(command, string_types):
        command = [command]

    # timeout != None is supported only for Unix
    if timeout is not None or (flush_stdout==False and flush_stderr==False):
        import subprocess32
        p = subprocess32.Popen(
            command, stdout=subprocess32.PIPE, stderr=subprocess32.PIPE, cwd=cwd, shell=True
        )

        assert flush_stdout == False, "Not supported flush_stdout"

        try:
            stdoutdata, stderrdata = p.communicate(timeout=timeout)
        except subprocess32.TimeoutExpired:
            return [], [], "timeout"

        return stdoutdata.split("\n") if stdoutdata else [], \
               stderrdata.split("\n") if stderrdata else [], \
               p.returncode
    else:
        p = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,  cwd=cwd, shell=True
        )

        out_thread = threading.Thread(name='stdout_thread', target=stdout_thread, args=(p, flush_stdout))
        err_thread = threading.Thread(name='stderr_thread', target=stderr_thread, args=(p, flush_stderr))

        err_thread.start()
        out_thread.start()

        out_thread.join()
        err_thread.join()

        p.wait()

        return _stdout_log.split("\n"), _stderr_log.split("\n"), p.returncode