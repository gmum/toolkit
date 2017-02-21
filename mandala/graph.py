
from igraph import Graph
from mandala import get_backend, BACKEND_PATH
import os

def sync_graph(func):

    def with_loaded_graph(*args, **kwargs):
        if len(args) > 0:
            graph_path = args[0]
        else:
            graph_path = kwargs['graph']

        if os.path.exists(graph_path):
            with open(graph_path, 'r') as f:
                graph = Graph.Read_Pickle(f)

            if 'graph' in kwargs.keys():
                del kwargs['graph']

            return func(graph, *args[1:], **kwargs)
        else:
            raise IOError("Graph file not found!")

    return with_loaded_graph


class DataNode(object):

    def __init__(self, node_id):
        self._id = node_id

    def eval(self):
        backend = get_backend(which='storage')(BACKEND_PATH)
        return backend.load(self._id)


def initialize_graph(path):
    if not os.path.exists(path):
        graph = Graph()
        graph.vs['name'] = []
        graph.vs['func_name'] = []
        graph.vs['func_path'] = []
        graph.vs['output_index'] = []

        with open(path, 'w') as f:
            graph.write_pickle(f)


@sync_graph
def add_node(graph, input_nodes_ids, func_name, func_path, output_index=0):

    assert isinstance(input_nodes_ids, list)

    ### check if output node exists
    # get succesors of input nodes
    successors = [graph.vs.find(name=node).successors() for node in input_nodes_ids]
    # intersect succesors
    candidate_nodes = reduce(lambda x, y: set(x).intersection(set(y)), successors, set(graph.vs['name']))

    # filter over arguments
    arg_filter = lambda x: graph.vs['func_name'] == func_name and \
                           graph.vs['func_path'] == func_path and \
                           graph.vs['output_index'] == output_index
    candidate_nodes = filter(arg_filter, candidate_nodes)
    # filter over predecessors number
    pred_filter = lambda x: len(graph.vs.find(name=x).predecessors()) == len(input_nodes_ids)
    candidate_nodes = filter(pred_filter, candidate_nodes)
    # check for graph corruption
    if len(candidate_nodes) > 1:
        raise ValueError("Corrupted graph! Found more than 1 existing candidate for a node!")
    elif len(candidate_nodes) == 1:
        return DataNode(candidate_nodes[0])

    node_id = str(int(graph.vs['name'][-1]) + 1) if len(graph.vs) > 0 else '0'

    # add new node to the graph
    graph.add_vertex(name=node_id, func_name=func_name, func_path=func_path, output_index=output_index)

    # connect that node to input nodes
    for parent_id in input_nodes_ids:
        graph.add_edge(parent_id, node_id)

    return DataNode(node_id)

@sync_graph
def find_nodes(graph, input_nodes_ids, func_name, func_path):
    # get succesors of input nodes
    try:
        successors = [graph.vs.find(name=node._id).successors() for node in input_nodes_ids]
    except:
        import pdb; pdb.set_trace()
    # intersect succesors
    candidate_nodes = reduce(lambda x, y: set(x).intersection(set(y)), successors, set(graph.vs['name']))

    # filter over arguments
    arg_filter = lambda x: graph.vs['func_name'] == func_name and \
                           graph.vs['func_path'] == func_path
    candidate_nodes = filter(arg_filter, candidate_nodes)
    # filter over predecessors number
    pred_filter = lambda x: len(graph.vs.find(name=x).predecessors()) == len(input_nodes_ids)
    candidate_nodes = filter(pred_filter, candidate_nodes)
    # check for graph corruption
    output_ids = [graph.vs.find(name=node_id)['output_index'] for node_id in candidate_nodes]

    tuples = sorted(zip(output_ids, candidate_nodes))
    if not [t[0] for t in tuples] == range(len(tuples)):
        raise ValueError("Corrupted graph! Lost some output node!")

    return tuple([DataNode(t[1]) for t in tuples])


def add_basic_type_node(graph_path, output_index):

    return add_node(graph_path,
                    input_nodes_ids=[],
                    func_name='_load_internal_cache',
                    func_path='mandala.internal_cache',
                    output_index=output_index)


def load(path):
    with open(path, 'r') as f:
        graph = Graph.Read_Pickle(f)
    return graph


def save(graph, path):
    with open(path, 'w') as f:
        graph.write_pickle(f)




