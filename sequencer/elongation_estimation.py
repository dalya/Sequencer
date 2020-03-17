#######################################################################################################################
####################################################### IMPORTS #######################################################
#######################################################################################################################

import numpy
import time
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def return_start_index_from_MST(graph):
    """Function returns the starting point of the sequence, which is defined as the least central node in the given graph.
    The least central node in the graph is defined using the closeness centrality.
    
    Parameters
    -------
    :param graph: networkx.classes.graph.Graph(), the graph that represents the Mininun Spanning Tree

    Returns
    -------
    :param start_index: integer, the index of the node found to be the starting point
    """
    centrality = nx.closeness_centrality(graph)
    indices = numpy.fromiter(centrality.keys(), dtype=int)
    centrality_measure = numpy.fromiter(centrality.values(), dtype=float)
    start_index = indices[numpy.argmin(centrality_measure)]

    return start_index

def return_MST_elongation(distance_arr):
    """Function estimates the elongation of the MST that is described by the given distance array. The input distance 
    array represents the distances of each node in the graph from the root of the graph (the starting point). 
    Funciton calculates the elongation by dividing the half-length of the tree by the half-width.
    The half-width is calculated as the average width in every depth level, and the half-length is calculated as the
    average distance from the root.

    Parameters
    -------
    :param distance_arr: list, a list that described the distance of each node from the root of the tree

    Returns
    -------
    :param mst_elongation: float, the elongation of the MST
    """
    graph_half_length = numpy.average(distance_arr) 
    g_unique, counts = numpy.unique(distance_arr, return_counts=True)
    graph_half_width = numpy.average(counts) / 2.
    mst_elongation = float(graph_half_length) / float(graph_half_width) + 1 

    return mst_elongation

def apply_MST_and_return_MST_and_elongation(distance_matrix, return_elongation=True):
    """Function converts the distance matrix into a fully-conncted graph and calculates its Minimum Spanning Tree (MST).
    Function has an option to return the elongation of the resulting MST. 

    Parameters
    -------
    :param distance_matrix: numpy.ndarray(), the distance matrix that will be converted into an MST.
    :param return_elongation: boolean (default=True), whether to return the elongation of the resulting MST.

    Returns
    -------
    :param G: networkx.classes.graph.Graph(), the graph that represents the resulting MST.
    :param mst_elongation (optional): float, the elongation of the resulting MST.
    """
    assert type(distance_matrix) == numpy.ndarray, "distance matrix must be numpy.ndarray"
    assert len(distance_matrix.shape) == 2, "distance matrix must have 2 dimensions"
    assert distance_matrix.shape[0] == distance_matrix.shape[1], "distance matrix must be NxN matrix"
    assert (~numpy.isnan(distance_matrix)).all(), "distance matrix contains nan values"
    assert (~numpy.isneginf(distance_matrix)).all(), "distance matrix contains negative infinite values"
    assert (distance_matrix.round(5) >= 0).all(), "distance matrix contains negative values"
    assert (distance_matrix.diagonal() == 0).all(), "distance matrix must contain zeros in its diagonal"

    min_span_dist_mat = minimum_spanning_tree(csr_matrix(distance_matrix)).toarray()
    G = nx.from_scipy_sparse_matrix(minimum_spanning_tree(csr_matrix(distance_matrix)))
    if return_elongation:
        start_index = return_start_index_from_MST(G)
        distance_dict = nx.shortest_path_length(G, start_index)
        distance_arr = numpy.fromiter(distance_dict.values(), dtype=int)
        mst_elongation = return_MST_elongation(distance_arr)
        return G, mst_elongation
    else:
        return G