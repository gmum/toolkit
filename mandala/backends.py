import os
import cPickle

REPOSITORY_DIR = os.path.join(os.path.dirname(__file__), "..")
GRAPH_PATH = os.path.join(REPOSITORY_DIR, 'graph.pkl')

from igraph import Graph
from mandala.internal_cache import  _load_internal_cache
from mandala import get_backend as _get_backend

def get_backend():
    return {"pickle": PickleBackend}[_get_backend()]


class PickleBackend(object):

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def save(self, object, node):
        with open(os.path.join(self.root_dir, node._id + '.pkl'), 'w') as f:
            cPickle.dump(object, f)

    def load(self, node):
        node_path = os.path.join(self.root_dir, node._id + '.pkl')
        if os.path.exists(node_path):
            with open(node_path, 'r') as f:
                object = cPickle.load(f)
            return object
        else:
            with open(GRAPH_PATH, 'r') as f:
                graph = Graph.Read_Pickle(f)

            v = graph.vs.find(name=node._id)
            assert v['func_name'] == '_load_internal_cache'
            assert v['func_path'] == 'mandala.internal_cache'
            output_index = v['output_index']
            cache = _load_internal_cache()
            return cache[output_index]

