
import os
import json
from six import string_types
# from backends import PickleBackend
import logging

REPOSITORY_DIR = os.path.join(os.path.dirname(__file__), "..")
GRAPH_PATH = os.path.join(REPOSITORY_DIR, 'graph.pkl')
STORAGE_PATH = os.path.join(REPOSITORY_DIR, '.backends')
MANDALA_CACHE = os.path.join(REPOSITORY_DIR, '.mandala_cache')

CONFIG = {
    "storage_backend": None,
    "graph_backend": None
}


def set_graph_path(graph_path):
    global GRAPH_PATH
    GRAPH_PATH = graph_path

def set_storage_backend(backend):
    assert isinstance(backend, string_types), "Pass backend as a string"
    CONFIG["storage_backend"] = backend


def set_graph_backend(graph_backend):
    assert isinstance(graph_backend, string_types), "Pass backend as a string"
    CONFIG['graph_backend'] = graph_backend

def get_backend():
    return CONFIG["storage_backend"]


