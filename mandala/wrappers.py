
from graphs import graph, DataNode
from mandala.internal_cache import cached_value_to_index
from backends import backend
import inspect


def wrap(function): # TODO: add default mandala meat params

    def run_function(node_kwargs):
        # eval kwargs
        args = {key: node.eval() for key, node in node_kwargs}
        returned = function(**args)

        if not isinstance(returned, tuple):
            returned = [returned]
        return returned

    def _check_storage(nodes):
        return all([backend.exists(node._id) for node in nodes])

    def _check_cache(nodes):
        return all([node.exists_cache() for node in nodes])


    def wrapped(*args, **kwargs):

        # get function info
        members_dict = dict(inspect.getmembers(function))
        func_name = members_dict['__name__']
        func_path = members_dict['__module__']

        # get arguments info
        argspecs = inspect.getargspec(function)
        arg_names = argspecs.args

        # TODO: get mandala meta params

        # add args to kwargs
        for key, arg in zip(arg_names[:len(args)], args):
            assert key not in kwargs
            kwargs[key] = arg # FIXME: do not modify kwargs?

        # process arguments into nodes
        new_kwargs = {}
        for key, arg in new_kwargs.iteritems():
            if isinstance(arg, DataNode):
                new_kwargs[key] = arg
            else:
                output_index = cached_value_to_index(arg)
                new_kwargs['key'] = graph.add_basic_type_node(output_index=output_index)

        # find nodes
        output_nodes = graph.find_output_nodes(input_nodes_ids=new_kwargs,
                                               func_name=func_name,
                                               func_path=func_path)

        # no nodes found
        if output_nodes is None:
            returned = run_function(new_kwargs)
            # create nodes
            output_nodes = []
            for i in xrange(len(returned)):
                node = graph.add_node(new_kwargs, func_name, func_path, output_index=i)
                output_nodes.append(node)
        # nodes found
        else:
            assert len(output_nodes) > 0
            storage_exists = _check_storage(output_nodes)
            cache_exists = _check_cache(output_nodes)

            if cache_exists:
                returned = [node._get_from_cache() for node in output_nodes]
            elif storage_exists:
                returned = [backend.load(node._id) for node in output_nodes]
            else:
                returned = run_function(new_kwargs)

        assert output_nodes is not None
        assert len(output_nodes) > 0
        assert len(returned) > 0

        # TODO: check mandala meta params for what to do with results

        # TODO: save returned where mandala meta params says to save

        if len(output_nodes) > 1:
            return tuple(output_nodes)
        else:
            return output_nodes[0]


    return wrapped












