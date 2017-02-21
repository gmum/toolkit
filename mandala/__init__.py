
import os
import json

from graph import DataGraph
REPOSITORY_DIR = os.path.join(os.path.dirname(__file__), "..")


GRAPH_PATH = os.path.join(REPOSITORY_DIR, 'mandala_graph.pkl')
MANDALA_CACHE = os.path.join(REPOSITORY_DIR, '.mandala_cache')


def set_graph_path(graph_path):
    global GRAPH_PATH
    GRAPH_PATH = graph_path


