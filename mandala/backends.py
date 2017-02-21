import os
import cPickle


class PickleBackend(object):

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def save(self, object, node):
        with open(os.path.join(self.root_dir, node._id + '.pkl'), 'w') as f:
            cPickle.dump(object, f)

    def load(self, node):
        with open(os.path.join(self.root_dir, node._id + '.pkl'), 'r') as f:
            object = cPickle.load(f)
        return object