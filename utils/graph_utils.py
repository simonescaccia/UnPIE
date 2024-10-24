import numpy as np
import tensorflow as tf
from spektral.data import Graph
from spektral.data.utils import to_disjoint

import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)


def graph_seq_to_graph(graph_seq, num_nodes):
    '''
    Convert a sequence of graphs to a DisjointGraph
    '''        
    x_list = []
    a_list = []
    y_list = []
    for graph in graph_seq:
        x, a, y = graph.x, graph.a, graph.y
        x_list.append(x)
        a_list.append(a)
        y_list.append(y)
    x, a, _ = to_disjoint(x_list, a_list)

    graph = Graph(x=x, a=a, y=y_list[0])
    return graph

def graph_to_graph_seq(x, a):
    '''
    Convert a DisjointGraph to a sequence of graphs
    - Graphs are undirected, so the adjacency matrix is symmetric
    '''
    graph_seq_batch = []
    i = 1 # last node of the current graph
    j = 0 # first node of the current graph
    for i in range(a.shape[1]):
        graph_seq = []
        while i < a.shape[1]:
            row = a[i, :i]
            col = a[:i, i]
            if np.sum(row) == 0 and np.sum(col) == 0:
                graph_seq.append((x[j:i], a[j:i, j:i]))
                j = i
            i += 1
        graph_seq_batch.append(graph_seq)
    graph_seq_batch = np.array(graph_seq_batch)
    return graph_seq_batch