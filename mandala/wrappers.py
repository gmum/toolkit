
from mandala import GRAPH_PATH, BACKEND_PATH
from graph import *
from mandala.internal_cache import cached_value_to_index
from mandala import get_backend
import inspect


def wrap(function):

    def wrapped(*args, **kwargs):

        backend = get_backend(which='storage')(BACKEND_PATH)
        new_args = []
        # TODO: remember about kwargs
        for arg in args:
            if isinstance(arg, DataNode):
                new_args.append(arg)
            else:
                output_index = cached_value_to_index(arg)
                new_args.append(add_basic_type_node(GRAPH_PATH, output_index=output_index))

        members_dict = dict(inspect.getmembers(function))
        func_name = members_dict['__name__']
        func_path = members_dict['__module__']

        output_nodes = find_nodes(GRAPH_PATH,
                                  input_nodes_ids=new_args,
                                  func_name=func_name,
                                  func_path=func_path)

        if len(output_nodes) > 0:
            return output_nodes
        else:
            values = [backend.load(node) for node in new_args]

            ret = function(*values)

            if isinstance(ret, tuple):
                results = []
                for i in xrange(len(ret)):
                    node = add_node(GRAPH_PATH, new_args, func_name, func_path, output_index=i)
                    backend.save(ret[i], node)
                    results.append(node)
                return tuple(results)
            else:
                node = add_node(GRAPH_PATH, new_args, func_name, func_path, output_index=0)
                backend.save(ret, node)
                return node

    return wrapped








