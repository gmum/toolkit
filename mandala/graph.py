
from igraph import Graph


def DataNode(object):

    def __init__(self, node_id):
        self.node_id = node_id


@sync_graph
def add_node(graph, input_nodes_ids, func_name, func_path, output_index=0):

    assert isinstance(input_nodes_ids, list)

    # check if output node exists
    if len(input_nodes_ids) > 0:

        # intersect nastepniki input nołdów if len(input_nodes_ids) > 0 else wez wszystkie nołdy
        # filter je po func_name, func_path, output_index
        # filtruj po liczbie poprzednikow
        # krzycz jesli zostalo wiecej niz 1 -> graph currupted
        # jesli jest zostal 1 to zwroc jego DataNode

    node_id = str(int(graph.vs['name'][-1]) + 1)

    # add new node to the graph
    graph.add_vertex(name=node_id, func_name=func_name, func_path=func_path, output_index=output_index)

    # connect that node to input nodes
    ca


    return DataNode(node_id)

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






def sync_graph(func):
    pass