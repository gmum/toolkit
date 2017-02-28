
from abc import ABCMeta, abstractmethod
import os
import cPickle


from . import CONFIG, STORAGE_PATH

# TODO: create abstraction BaseBacked wtih save, load, exists, search_by_value (only for "magic" nodes?)
# TODO: create JSONBackend

class PickleBackend(object):

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def _get_path(self, node_id):
        return os.path.join(self.root_dir, node_id + '.pkl')

    def save(self, object, node_id):
        # TODO: check if already exists and throw error if it does
        with open(os.path.join(self.root_dir, node_id + '.pkl'), 'w') as f:
            cPickle.dump(object, f)

    def load(self, node_id):
        node_path = os.path.join(self.root_dir, node_id + '.pkl')
        if os.path.exists(node_path):
            with open(node_path, 'r') as f:
                object = cPickle.load(f)
            return object
        else:
            # TODO: throw custom backend error
            raise IOError("Backend file not found")

    def get_abs_path(self, node_id):
        return os.path.join(self.root_dir, node_id + '.pkl')

    def exists(self, node_id):
        return os.path.exists(self._get_path(node_id))


# set main storage
backend = None
if CONFIG['storage_backend'] == 'pickle':
    backend = PickleBackend(STORAGE_PATH)
else:
    raise ValueError("Wrong SORAGE backend: {}".format(CONFIG['storage_backend']))