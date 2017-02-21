
from mandala import GRAPH_PATH
from graph import *
from mandala.internal_cache import _cached_value_to_index

def wrap(function):

    def wrapper(*args, **kwargs):

        new_args = []
        # TODO: remember aboyt kwargs
        for arg in args:
            if isinstance(arg, DataNode):
                new_args.append(arg)
            else:
                output_index = _cached_value_to_index(arg)
                new_args.append(add_basic_type_node(GRAPH_PATH, output_index=output_index))

        #for key, value in kwargs.iteritems():






