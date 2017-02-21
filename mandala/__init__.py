
import os
import json
from six import string_types
from backends import *
import logging

REPOSITORY_DIR = os.path.join(os.path.dirname(__file__), "..")


GRAPH_PATH = os.path.join(REPOSITORY_DIR, 'mandala_graph.pkl')
BACKEND_PATH = os.path.join(REPOSITORY_DIR, 'mandala_backends')
MANDALA_CACHE = os.path.join(REPOSITORY_DIR, '.mandala_cache')

BACKENDS = {'storage': None}


def set_graph_path(graph_path):
    global GRAPH_PATH
    GRAPH_PATH = graph_path

def set_storage_backend(backend):
    assert isinstance(backend, string_types), "Pass backend as a string"
    global BACKENDS
    backends = {'pickle': PickleBackend}
    BACKENDS['storage'] = backends[backend]
    logging.info("Set storage backend as {}".format(backend))

def get_backend(which):
    return BACKENDS[which]