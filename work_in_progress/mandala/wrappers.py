
from .graphs import graph, DataNode
from .internal_cache import cached_value_to_index
from .backends import backend
import inspect

import pdb


class mandala(object):

    def __init__(self, store=True, cache=False):

        self.save_storage = store
        self.save_cache = cache
        self.function = None

    def run_function(self, node_kwargs):
        # eval kwargs
        args = {key: node.eval() for key, node in node_kwargs.iteritems()}
        returned = self.function(**args)

        if not isinstance(returned, tuple):
            returned = [returned]
        return returned

    def _check_storage(self, nodes):
        return all([backend.exists(node._id) for node in nodes])

    def _check_cache(self, nodes):
        return all([node.exists_cache() for node in nodes])

    def _save_results(self, where, results):
        assert len(results) > 0
        if where == 'storage':
            for node_id, ret in results.iteritems():
                backend.save(ret, node_id)
        elif where == 'cache':
            for node_id, ret in results.iteritems():
                DataNode(node_id).cache_data(ret) # FIXME: ugly? bad?


    def __call__(self, function):

        self.function = function

        def wrapped(*args, **kwargs):

            # get function info
            members_dict = dict(inspect.getmembers(function))
            func_name = members_dict['__name__']
            func_path = members_dict['__module__']

            # get arguments info
            argspecs = inspect.getargspec(function)
            arg_names = argspecs.args

            # get mandala meta params
            if 'meta_mandala' in kwargs.keys():
                meta_params = kwargs['meta_mandala']

                self.save_storage = meta_params.get('store', self.save_storage)
                self.save_cache = meta_params.get('cache', self.save_cache)

            # add args to kwargs
            for key, arg in zip(arg_names[:len(args)], args):
                assert key not in kwargs
                kwargs[key] = arg # FIXME: do not modify kwargs?

            # process arguments into nodes
            new_kwargs = {}
            for key, arg in kwargs.iteritems():
                if isinstance(arg, DataNode):
                    new_kwargs[key] = arg
                else:
                    output_index = cached_value_to_index(arg)
                    new_kwargs[key] = graph.add_basic_type_node(output_index=output_index)

            # find nodes
            output_nodes = graph.find_output_nodes(input_nodes_dict=new_kwargs,
                                                   func_name=func_name,
                                                   func_path=func_path)

            results = {}
            # no nodes found
            if output_nodes is None:
                returned = self.run_function(new_kwargs)
                # create nodes
                output_nodes = []
                for i in xrange(len(returned)):
                    node = graph.add_node(new_kwargs, func_name, func_path, output_index=i)
                    output_nodes.append(node)
                    results[node._id] = returned[i]

                # save results as requested by mandala meta
                if self.save_storage:
                    self._save_results(where='storage', results=results)
                if self.save_cache:
                    self._save_results(where='cache', results=results)
            # nodes found
            else:
                assert len(output_nodes) > 0
                # check if results are stored somewhere
                storage_exists = self._check_storage(output_nodes)
                cache_exists = self._check_cache(output_nodes)

                # results not in storage and should be
                if not storage_exists and self.save_storage:
                    # check if results are in cache
                    if cache_exists: # results are in cache
                        results = {node._id: node._get_from_cache() for node in output_nodes}
                    # there are no saved results, calculate them
                    else:
                        returned = self.run_function(new_kwargs)
                        results = {node._id: ret for node, ret in zip(output_nodes, returned)}
                    # save to storge
                    self._save_results(where='storage', results=results)
                # results are not in cache and should be
                elif not cache_exists and self.save_cache:
                    # check if results are in storage
                    if storage_exists:
                        results = {node._id: backend.load(node._id) for node in output_nodes}
                    # there are no saved results, calculate them
                    else:
                        returned = self.run_function(new_kwargs)
                        results = {node._id: ret for node, ret in zip(output_nodes, returned)}
                    # save to cache0
                    self._save_results(where='cache', results=results)

            if len(output_nodes) > 1:
                return tuple(output_nodes)
            else:
                return output_nodes[0]

        return wrapped












