
from graphs import graph, DataNode
from mandala.internal_cache import cached_value_to_index
from backends import backend
import inspect


def wrap(function):

    def wrapped(*args, **kwargs):

        new_args = []
        # TODO: remember about kwargs
        for arg in args:
            if isinstance(arg, DataNode):
                new_args.append(arg)
            else:
                print "debug type:", arg
                output_index = cached_value_to_index(arg)
                new_args.append(graph.add_basic_type_node(output_index=output_index))

        members_dict = dict(inspect.getmembers(function))
        func_name = members_dict['__name__']
        func_path = members_dict['__module__']

        output_nodes = graph.find_nodes(
                                  input_nodes_ids=new_args,
                                  func_name=func_name,
                                  func_path=func_path)

        if len(output_nodes) > 0:
            return tuple(output_nodes) if len(output_nodes) > 1 else output_nodes[0]
        else:
            values = [backend.load(node) for node in new_args]

            ret = function(*values)

            if isinstance(ret, tuple):
                results = []
                for i in xrange(len(ret)):
                    node = graph.add_node(new_args, func_name, func_path, output_index=i)
                    backend.save(ret[i], node)
                    results.append(node)
                return tuple(results)
            else:
                node = graph.add_node(new_args, func_name, func_path, output_index=0)
                backend.save(ret, node)
                print node
                return node

    return wrapped








