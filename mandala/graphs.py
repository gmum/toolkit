from abc import ABCMeta, abstractmethod
from igraph import Graph
import os
from collections import defaultdict

from mandala import GRAPH_PATH, CONFIG
from backends import backend

import pdb


class DataNode(object):

    _cache_storage = {}
    _count_references = defaultdict(int)

    def __init__(self, _id):
        self._id = _id

    @classmethod
    def _increment_reference(cls, _id):
        cls._count_references[_id] += 1

    @classmethod
    def _decrement_reference(cls, _id):
        cls._count_references[_id] -= 1
        assert  cls._count_references[_id] >= 0, "Node cache corrupted"

        if cls._count_references[_id] == 0 and _id in cls._cache_storage:
            del cls._cache_storage[_id]

    def cache_data(self, data):
        self.__class__._cache_storage[self._id] = data

    def _get_from_cache(self):
        return self.__class__._cache_storage[self._id]

    def exists_cache(self):
        return self._id in self.__class__._cache_storage

    def eval(self):
        # try loading from storage
        try:
            return backend.load(self._id)
        except IOError: # TODO: excpet custom backend error in future
            pass

        # try getting data from class cache
        try:
            return self._get_from_cache()
        except KeyError:
            pass

        # get node from graph
        func_name, func_path, output_index = graph.get_node_func_info(self._id)
        exec ('from {} import {}'.format(func_path, func_name))
        function = eval(func_name)

        # TODO: get parent nodes from graph
        node_kwargs = # get parent nodes
        # TODO: run it
        ret = function(**node_kwargs)

        return ret


class BaseGraph(object):
    __metaclass__ = ABCMeta

    def __init__(self, path):
        self.path = path
        self.graph = None
        self.initialize()


    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self):
        raise NotImplementedError()

    @abstractmethod
    def load(self):
        raise NotImplementedError()

    @abstractmethod
    def _find_output_nodes(self, input_nodes_ids, func_name, func_path):
        raise NotImplementedError()

    def find_output_nodes(self, **kwargs):
        self.load()
        return self._find_output_nodes(**kwargs)

    @abstractmethod
    def _add_node(self, input_nodes_ids, func_name, func_path, output_index=0):
        raise NotImplementedError()

    def add_node(self, *args, **kwargs):
        self.load()
        node = self._add_node(*args, **kwargs)
        self.save()
        return node

    def add_basic_type_node(self, output_index):
        return self.add_node(input_nodes_ids=[],
                             func_name='_load_internal_cache',
                             func_path='mandala.internal_cache',
                             output_index=output_index)

    def get_node_func_info(self, node_id):
        self.load()
        return self._get_node_func_info(node_id)

    @abstractmethod
    def _get_node_func_info(self, node_id):
        raise NotImplementedError()


class iGraphGraph(BaseGraph):

    def initialize(self):
        if not os.path.exists(self.path):
            graph = Graph(directed=True)
            graph.vs['name'] = []
            graph.vs['func_name'] = []
            graph.vs['func_path'] = []
            graph.vs['output_index'] = []

            with open(self.path, 'w') as f:
                graph.write_pickle(f)

    def save(self):
        with open(self.path, 'w') as f:
            self.graph.write_pickle(f)

    def load(self):
        with open(self.path, 'r') as f:
            self.graph = Graph.Read_Pickle(f)

    def _find_output_nodes(self, input_nodes_dict, func_name, func_path):
        # get succesors of input nodes
        successors = [[v["name"] for v in self.graph.vs.find(name=node._id).successors()] for node in input_nodes_dict.values()]
        # intersect succesors
        candidate_nodes = reduce(lambda x, y: set(x).intersection(set(y)), successors, set(self.graph.vs["name"]))
        # filter over arguments
        arg_filter = lambda x: self.graph.vs.find(name=x)['func_name'] == func_name and \
                               self.graph.vs.find(name=x)['func_path'] == func_path
        candidate_nodes = filter(arg_filter, candidate_nodes)
        # filter over predecessors number
        pred_filter = lambda x: len(self.graph.vs.find(name=x).predecessors()) == len(input_nodes_dict)
        candidate_nodes = filter(pred_filter, candidate_nodes)
        if len(candidate_nodes) == 0:
            return None

        # check for graph corruption
        output_ids = [self.graph.vs.find(name=node_id)['output_index'] for node_id in candidate_nodes]

        tuples = sorted(zip(output_ids, candidate_nodes))
        if not [t[0] for t in tuples] == range(len(tuples)):
            raise ValueError("Corrupted graph! Lost some output node!")

        return [DataNode(t[1]) for t in tuples]

    def _add_node(self, input_nodes_dict, func_name, func_path, output_index=0):

        assert isinstance(input_nodes_dict, dict)

        ### check if output node exists
        # get succesors of input nodes
        successors = [self.graph.vs.find(name=node._id).successors() for node in input_nodes_dict.values()]
        # intersect succesors
        candidate_nodes = reduce(lambda x, y: set(x).intersection(set(y)), successors, set(self.graph.vs['name']))

        # filter over arguments
        arg_filter = lambda x: self.graph.vs['func_name'] == func_name and \
                               self.graph.vs['func_path'] == func_path and \
                               self.graph.vs['output_index'] == output_index
        candidate_nodes = filter(arg_filter, candidate_nodes)
        # filter over predecessors number
        pred_filter = lambda x: len(self.graph.vs.find(name=x).predecessors()) == len(input_nodes_dict)
        candidate_nodes = filter(pred_filter, candidate_nodes)
        # check for graph corruption
        if len(candidate_nodes) > 1:
            raise ValueError("Corrupted graph! Found more than 1 existing candidate for a node!")
        elif len(candidate_nodes) == 1:
            return DataNode(candidate_nodes[0])

        node_id = str(int(self.graph.vs['name'][-1]) + 1) if len(self.graph.vs) > 0 else '0'

        # add new node to the graph
        self.graph.add_vertex(name=node_id, func_name=func_name, func_path=func_path, output_index=output_index)

        # connect that node to input nodes
        for arg_name, parent_id in input_nodes_dict.iteritems():
            self.graph.add_edge(parent_id._id, node_id, name=arg_name)

        return DataNode(node_id)

    def _find_node(self, node_id):
        return self.graph.vs.find(name=node_id)

    def _get_node_func_info(self, node_id):
        node = self._find_node(node_id)
        return node['func_name'], node['func_path'], node['output_index']




# set main graph
graph = None
if CONFIG['graph_backend'] == 'iGraph':
    graph = iGraphGraph(GRAPH_PATH)
else:
    raise ValueError("Wrong graph backend: {}".format(CONFIG['graph_backend']))
