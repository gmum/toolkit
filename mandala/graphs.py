from abc import ABCMeta, abstractmethod
from igraph import Graph
import os

from mandala import GRAPH_PATH, CONFIG
from backends import backend

import pdb


# TODO: fix this class?
class DataNode(object):

    def __init__(self, node_id):
        self._id = node_id

    def eval(self):
        return backend.load(self._id)

    def eeeeeeval(self):
        try:
            return self.eval()
        except IOError: # TODO: code and use custom BackendError
            pass
            # load graph
            # get vertex for self._id
            # get all predeccessors


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
    def _find_nodes(self, input_nodes_ids, func_name, func_path):
        raise NotImplementedError()

    def find_nodes(self, **kwargs):
        self.load()
        return self._find_nodes(**kwargs)

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

    def _find_nodes(self, input_nodes_ids, func_name, func_path):
        # get succesors of input nodes
        successors = [[v["name"] for v in self.graph.vs.find(name=node._id).successors()] for node in input_nodes_ids]
        # intersect succesors
        candidate_nodes = reduce(lambda x, y: set(x).intersection(set(y)), successors, set(self.graph.vs["name"]))
        # filter over arguments
        arg_filter = lambda x: self.graph.vs.find(name=x)['func_name'] == func_name and \
                               self.graph.vs.find(name=x)['func_path'] == func_path
        candidate_nodes = filter(arg_filter, candidate_nodes)
        # filter over predecessors number
        pred_filter = lambda x: len(self.graph.vs.find(name=x).predecessors()) == len(input_nodes_ids)
        candidate_nodes = filter(pred_filter, candidate_nodes)
        # check for graph corruption
        output_ids = [self.graph.vs.find(name=node_id)['output_index'] for node_id in candidate_nodes]

        tuples = sorted(zip(output_ids, candidate_nodes))
        if not [t[0] for t in tuples] == range(len(tuples)):
            raise ValueError("Corrupted graph! Lost some output node!")

        return tuple([DataNode(t[1]) for t in tuples])

    def _add_node(self, input_nodes_ids, func_name, func_path, output_index=0):

        assert isinstance(input_nodes_ids, list)

        ### check if output node exists
        # get succesors of input nodes
        successors = [self.graph.vs.find(name=node._id).successors() for node in input_nodes_ids]
        # intersect succesors
        candidate_nodes = reduce(lambda x, y: set(x).intersection(set(y)), successors, set(self.graph.vs['name']))

        # filter over arguments
        arg_filter = lambda x: self.graph.vs['func_name'] == func_name and \
                               self.graph.vs['func_path'] == func_path and \
                               self.graph.vs['output_index'] == output_index
        candidate_nodes = filter(arg_filter, candidate_nodes)
        # filter over predecessors number
        pred_filter = lambda x: len(self.graph.vs.find(name=x).predecessors()) == len(input_nodes_ids)
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
        for parent_id in input_nodes_ids:
            self.graph.add_edge(parent_id._id, node_id)

        return DataNode(node_id)


# set main graph
graph = None
if CONFIG['graph_backend'] == 'iGraph':
    graph = iGraphGraph(GRAPH_PATH)
else:
    raise ValueError("Wrong graph backend: {}".format(CONFIG['graph_backend']))
