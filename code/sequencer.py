#######################################################################################################################
####################################################### IMPORTS #######################################################
#######################################################################################################################

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy
import glob
#from emd import emd
import time
from astropy.io import fits
import os
import shutil
import pickle
import networkx as nx

from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
from scipy.stats import wasserstein_distance
from scipy.stats import energy_distance
from scipy.stats import entropy
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr, pearsonr
from scipy.signal import medfilt
from scipy.interpolate import interp1d

# parallelization functions
from joblib import Parallel, delayed, dump, load
import multiprocessing


#######################################################################################################################
###################################################### DATA I/O #######################################################
#######################################################################################################################

def load_distance_matrices_from_path(path, scale_list, estimator_list):
    """
    function loads the distance matrices object into memory and checks if this object correspnds to the scale list and estimator list given
    """
    input_file = open(path, "rb")
    distance_matrix_dictionary = pickle.load(input_file)
    input_file.close()

    distance_matrix_dictionary_small = {}
    for estimator_index, estimator_value in enumerate(estimator_list):
        scale_list_for_estimator = scale_list[estimator_index]
        for scale_value in scale_list_for_estimator:
            if (estimator_value, scale_value) not in list(distance_matrix_dictionary.keys()):
                print("the scale list does not correspond to the distance matrices loaded")
                return 0
            else:
                distance_matrix_dictionary_small[(estimator_value, scale_value)] = distance_matrix_dictionary[(estimator_value, scale_value)]
    return distance_matrix_dictionary_small

def dump_distance_matrices_to_path(path, distance_matrix_dictionary):
    """
    function dumps the distance matrix dictionary to the given path, for future use
    """
    output_file = open(path, "wb")
    pickle.dump(distance_matrix_dictionary, output_file)
    output_file.close()

#######################################################################################################################
################################################# Distance measures ###################################################
#######################################################################################################################

def return_distance_matrix(grid, objects_list, estimator, choice_parallelization):
    """
    Function estimates the distance matrix of the given objects with the given estimator.

    @grid: array-like, the grid on which the objects are interpolated (the x-axis).
    @objects_list: array-like, the list of objects with their measured features.
    """
    grid = numpy.array(grid)
    objects_list = numpy.array(objects_list)

    assert ((len(grid.shape) == 1) or (len(grid.shape) == 2)), "objects can be 1 or 2 dimensional"
    assert (~numpy.isnan(grid)).all(), "grid cannot contain nan values"
    assert (~numpy.isinf(grid)).all(), "grid cannot contain infinite values"
    assert (~numpy.isneginf(grid)).all(), "grid cannot contain negative infinite values"
    assert (~numpy.isnan(objects_list)).all(), "objects_list cannot contain nan values"
    assert (~numpy.isinf(objects_list)).all(), "objects_list cannot contain infinite values"
    assert (~numpy.isneginf(objects_list)).all(), "objects_list cannot contain negative infinite values"
    if len(grid.shape) == 1:
        assert (grid.shape[0] == objects_list.shape[1]), "the grid and the objects must have the same dimensions"
    if len(grid.shape) == 2:
        assert ((grid.shape[0] == objects_list.shape[1]) and (grid.shape[1] == objects_list.shape[2])), "the grid and the objects must have the same dimensions"
    assert estimator in ['EMD_brute_force', 'energy', 'KL', 'L2'], "the distance estimator must be: EMD_brute_force, energy, KL, or L2"
    if estimator == "EMD_brute_force":
        assert (objects_list >= 0).all(), "the EMD distance can only be applied to non-negative values"
    if estimator == "energy":
        assert (objects_list >= 0).all(), "the energy distance can only be applied to non-negative values"
    if (estimator == "KL") and (len(grid.shape) == 1):
        assert (objects_list > 0).all(), "the KL distance can only be applied to positive values"

    # if the distance claculation should be carried out in parallel, save the data that will be used
    if choice_parallelization:
        folder = './joblib_memmap'
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        data_filename_memmap = os.path.join(folder, 'data_memmap')
        dump(objects_list, data_filename_memmap)
        objects_list = load(data_filename_memmap, mmap_mode='r')

        grid_filename_memmap = os.path.join(folder, 'grid_memmap')
        dump(grid, grid_filename_memmap)
        grid = load(grid_filename_memmap, mmap_mode='r')

    if estimator == "EMD_brute_force":
        distance_matrix = return_emd_mat_brute_force(grid, objects_list, choice_parallelization)

    if estimator == "energy":
        distance_matrix = return_energy_mat(grid, objects_list, choice_parallelization)

    if estimator == "KL":
        distance_matrix = return_kl_mat(objects_list, choice_parallelization)

    if estimator == "L2":
        distance_matrix = return_L2_mat(objects_list, choice_parallelization)

    return distance_matrix

#######################################################################################################################
################################################## Scale functions ####################################################
#######################################################################################################################

def normalise_objects(objects_list):
    """
    function normalises the objects such that the sum of each object is 1
    objects_list: a numpy array of 2 or 3 dimensions
    """
    assert ((len(objects_list.shape) == 2) or (len(objects_list.shape) == 3)), "objects can be either 1D or 2D"

    if len(objects_list.shape) == 2:
        sum_vector = numpy.sum(objects_list, axis=1)
        objects_list_normalised = objects_list / sum_vector[:, numpy.newaxis]

    if len(objects_list.shape) == 3:
        sum_vector = numpy.sum(objects_list, axis=(1,2))
        objects_list_normalised = objects_list / sum_vector[:, numpy.newaxis, numpy.newaxis]

    return objects_list_normalised

def divide_to_chunks_1D(grid, objects_list, N_chunks):
    """
    The function divides the data into chunks according to N_chunks. 
    The function then normalizes each split chunk to have a sum of 1, which is required by the distance metrics.
    The data is assumed to be 1-dimensional.
    The function uses numpy.array operations only partialy (which makes it slower), because the split arrays
    can have different sizes. This is because N_chunks can be an integer that does not equally divide the axis.

    @grid: the grid on which the objects are interpolated, a vector in this case.
    @objects_list: the list of objects with their measured features, assumed to be a 2D array.
    @N_chunks: an integer, the number of chunks to which to divide the data.
    """
    grid_split = numpy.array_split(grid, N_chunks)
    objects_list_split = numpy.array_split(objects_list, N_chunks, axis=-1)
    objects_list_split_normalised = []
    for objects_list_chunk in objects_list_split:
        sum_vec = numpy.sum(objects_list_chunk, axis=-1)
        object_list_chunk_norm = objects_list_chunk / sum_vec[:, numpy.newaxis]
        objects_list_split_normalised.append(object_list_chunk_norm)
            
    return grid_split, objects_list_split_normalised

def divide_to_chunks_2D(grid, objects_list, N_chunks):
    """
    The function divides the data into chunks according to N_chunks. 
    The function then normalizes each split chunk to have a sum of 1, which is required by the distance metrics.
    The data is assumed to be 2-dimensional.
    The function uses numpy.array operations only partialy (which makes it slower), because the split arrays
    can have different sizes. This is because N_chunks can be an integer that does not equally divide the axis.

    @grid: the grid on which the objects are interpolated, a matrix in this case.
    @objects_list: the list of objects with their measured features, assumed to be a 3D array.
    @N_chunks: a tuple of length 2, the number of chunks to which to divide the data.
    """
    grid_split = []
    for grid_split_tmp in numpy.array_split(grid, N_chunks[0], axis=0):
        grid_split += numpy.array_split(grid_split_tmp, N_chunks[1], axis=1)
    
    objects_list_split_normalised = []
    objects_list_split_1 = numpy.array_split(objects_list, N_chunks[0], axis=1)
    for objects_list_split_tmp in objects_list_split_1:
        objects_list_split_2 = numpy.array_split(objects_list_split_tmp, N_chunks[1], axis=2)
        for objects_list_split in objects_list_split_2:
            sum_vec = numpy.sum(objects_list_split, axis=(1,2))
            objects_list_split_normalised.append(objects_list_split / sum_vec[:, numpy.newaxis, numpy.newaxis])
            
    return grid_split, objects_list_split_normalised

def divide_to_chunks(grid, objects_list, N_chunks):
    """
    The main function that divides the data into chuncks. The function takes as input the grid and the object list
    and splits them into chunks according to N_chunks. The function is used to construct a multi-scale similarity measures.
    The function does not assume that N_chunks can divide the data into equaly-sized chunks.

    @grid: the grid on which the objects are interpolated (the x axis of the information)
    @objects_list: the list of objects with their measured features
    @N_chunks: the number of chunks to which to divide the data
    """
    assert ((len(grid.shape) == 1) or (len(grid.shape) == 2)), "objects can be either 1D or 2D"

    if len(grid.shape) == 1:
        return divide_to_chunks_1D(grid, objects_list, N_chunks)
    if len(grid.shape) == 2:
        return divide_to_chunks_2D(grid, objects_list, N_chunks)

#######################################################################################################################
################################################## GRAPH FUNCTIONS ####################################################
#######################################################################################################################

def return_MST_axis_ratio(distance_arr):
    """
    Function takes as an input a list that contains the distace of each object from the root of the graph.
    Function measures the axis ratio of the MST, namely the helf-length of the tree divided by the half-width of the tree.
    The width of the tree is calculated as the average width in every depth level, and the length is calculated 
    as the average distance from the root.

    @distance_arr: a list that describes the distance of each node from the root of the tree, given 
    in number of nodes that separate between them. Therefore, the list consists of integer values and its
    size is similar to the length of the input data.

    WARNING: this function can only be ran within the function: apply_MST_and_return_BFS_DFS_ordering
    """
    graph_half_length = numpy.average(distance_arr) 
    g_unique, counts = numpy.unique(distance_arr, return_counts=True)
    graph_half_width = numpy.average(counts) / 2.
    mst_axis_ratio = graph_half_length / float(graph_half_width)

    return mst_axis_ratio

def return_start_index_from_MST(graph):
    """
    Function takes as an input a graph (by networkx) and returns a suggested starting point for the graph.
    The function uses "closeness centrality" to find the less central nose in the graph, which will serve as the starting index.
    
    @graph: networkx.classes.graph.Graph, the graph that represents the mininun spanning tree
    """
    centrality = nx.closeness_centrality(graph)
    indices = numpy.fromiter(centrality.keys(), dtype=int)
    centrality_measure = numpy.fromiter(centrality.values(), dtype=float)
    start_index = indices[numpy.argmin(centrality_measure)]

    return start_index

def apply_MST_and_return_MST_and_axis_ratio(distance_matrix, return_axis_ratio=True):
    """
    Function converts the distance matrix into a fully-conncted graph and calculates its minimum spanning tree (MST).
    Function has an option to return the axis ratio of the resulting MST. 

    @distance_matrix: 2D numpy array, the distance matrix to be converted into a minimum spanning tree.
    @return_axis_ratio: boolean, whether or not to return the axis ratio of the resulting MST, default is True.
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
    if return_axis_ratio:
        start_index = return_start_index_from_MST(G)
        distance_dict = nx.shortest_path_length(G, start_index)
        distance_arr = numpy.fromiter(distance_dict.values(), dtype=int)
        mst_axis_ratio = return_MST_axis_ratio(distance_arr)
        return G, mst_axis_ratio
    else:
        return G

def apply_MST_and_return_BFS_DFS_ordering(distance_matrix, start_index=None, return_axis_ratio=True, return_MST=True):
    """
    Function converts the distance matrix into a fully-connected graph and calculates its minimum spanning tree (MST).
    The function also returns two walks within the tree: BFS and DFS.
    The function also has an option to return the axis ratio of the resulting MST.

    @distance_matrix: 2D numpy array, the distance matrix to be converted into a minimum spanning tree.
    @start_index: the index in the matrix from which to start the BFS/DFS walk within the MST. 
                  If not specified, the function calculates a "good" place to start from using closeness centrality measure.
    @return_axis_ratio: boolean, whether or not to return the axis ratio of the resulting MST, default is True.
    @return_MST: boolean, whether or not to return the MST itself, default is True.
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
    if start_index == None:
        start_index = return_start_index_from_MST(G)
    # dfs
    ordering_dfs = numpy.fromiter(nx.dfs_preorder_nodes(G, start_index), dtype=int)
    # bfs
    distance_dict = nx.shortest_path_length(G, start_index)
    distance_inds = numpy.fromiter(distance_dict.keys(), dtype=int)
    distance_arr = numpy.fromiter(distance_dict.values(), dtype=int)
    ordering_bfs = distance_inds[numpy.argsort(distance_arr)]

    if return_axis_ratio and return_MST:
        mst_axis_ratio = return_MST_axis_ratio(distance_arr)
        return ordering_bfs, ordering_dfs, mst_axis_ratio, G
    elif return_axis_ratio:
        mst_axis_ratio = return_MST_axis_ratio(distance_arr)
        return ordering_bfs, ordering_dfs, mst_axis_ratio
    else:
        return ordering_bfs, ordering_dfs

#######################################################################################################################
######################################## PROXIMIY/DISTANCE matrix conversion ##########################################
#######################################################################################################################

def return_proximity_matrix_populated_by_MSTs_avg_prox(MST_list, weight_list, to_average_N_best_estimators=False, number_of_best_estimators=None):
    """
    Function populates all the given minimum spanning trees into a proximity matrix.
    Function weights the different edges based on the weight list which is also given.
    Cells in the matrix that do not correspond to any edge will be filled with zero (no proximity).
    Cells in the diagonal of the matrix will be filled with numpy.inf (infinite proximity).

    @MST_list: a list that contains all the MST objects, each of the objects is a networkx.classes.graph.Graph.
    @weight_list: a weight list that represents the relative importance of each MST. should be positive values.
    """
    assert type(MST_list) == list, "MST_list should be a list"
    assert (numpy.array(weight_list) >= 0).all(), "weights in weight_list should be non-negative"
    assert numpy.fromiter([type(mst) == nx.classes.graph.Graph for mst in MST_list], dtype=bool).all(), "MST_list should contain networkx.classes.graph.Graph objects"

    # start by taking only the best estimators, if specified
    if to_average_N_best_estimators:
        indices = numpy.argsort(weight_list)[::-1][:number_of_best_estimators]
        weight_list_new = []
        MST_list_new = []
        for index_good in indices:
            weight_list_new.append(weight_list[index_good])
            MST_list_new.append(MST_list[index_good])

        weight_list = list(weight_list_new)
        MST_list = list(MST_list_new)

    N = MST_list[0].number_of_nodes()
    sum_of_weights = numpy.sum(weight_list)
    proximity_matrix = numpy.zeros((N, N))

    for mst_index, mst in enumerate(MST_list):
        weight_of_mst = weight_list[mst_index]
        # go over all the edges and populate, this should be symmetric
        for edge in mst.edges():
            node1 = edge[0]
            node2 = edge[1]
            distance = mst[node1][node2]['weight']
            proximity_matrix[node1, node2] += (weight_of_mst * (1.0 / distance))
            proximity_matrix[node2, node1] += (weight_of_mst * (1.0 / distance))
    proximity_matrix /= sum_of_weights
    numpy.fill_diagonal(proximity_matrix, numpy.inf)
    return proximity_matrix

def return_proximity_matrix_populated_by_MSTs_avg_dist(MST_list, weight_list, to_average_N_best_estimators=False, number_of_best_estimators=None):
    """
    Function populates all the given minimum spanning trees into a proximity matrix.
    Function weights the different edges based on the weight list which is also given.
    Cells in the matrix that do not correspond to any edge will be filled with zero (no proximity).
    Cells in the diagonal of the matrix will be filled with numpy.inf (infinite proximity).

    @MST_list: a list that contains all the MST objects, each of the objects is a networkx.classes.graph.Graph.
    @weight_list: a weight list that represents the relative importance of each MST. should be positive values.
    """
    assert type(MST_list) == list, "MST_list should be a list"
    assert (numpy.array(weight_list) >= 0).all(), "weights in weight_list should be non-negative"
    assert numpy.fromiter([type(mst) == nx.classes.graph.Graph for mst in MST_list], dtype=bool).all(), "MST_list should contain networkx.classes.graph.Graph objects"

    # start by taking only the best estimators, if specified
    if to_average_N_best_estimators:
        indices = numpy.argsort(weight_list)[::-1][:number_of_best_estimators]
        weight_list_new = []
        MST_list_new = []
        for index_good in indices:
            weight_list_new.append(weight_list[index_good])
            MST_list_new.append(MST_list[index_good])

        weight_list = list(weight_list_new)
        MST_list = list(MST_list_new)

    N = MST_list[0].number_of_nodes()
    sum_of_weights = numpy.sum(weight_list)
    distance_matrix = numpy.zeros((N, N))

    for mst_index, mst in enumerate(MST_list):
        weight_of_mst = weight_list[mst_index]
        # go over all the edges and populate, this should be symmetric
        for edge in mst.edges():
            node1 = edge[0]
            node2 = edge[1]
            distance = mst[node1][node2]['weight']
            distance_matrix[node1, node2] += (weight_of_mst * distance)
            distance_matrix[node2, node1] += (weight_of_mst * distance)

    # now convert the distance matrix into a proximity matrix
    proximity_matrix = convert_distance_to_proximity_matrix(distance_matrix)
    return proximity_matrix


def convert_proximity_to_distance_matrix(proximity_matrix):
    """
    Function converts the given proximity matrix to a distance matrix and returns it.

    @proximity_matrix: a 2d numpy.array(), a matrix that represents the proximities between the objects.
    """
    assert type(proximity_matrix) == numpy.ndarray, "proximity matrix must be numpy.ndarray"
    assert len(proximity_matrix.shape) == 2, "proximity matrix must have 2 dimensions"
    assert proximity_matrix.shape[0] == proximity_matrix.shape[1], "proximity matrix must be NxN matrix"
    assert (~numpy.isnan(proximity_matrix)).all(), "proximity matrix contains nan values"
    assert (~numpy.isneginf(proximity_matrix)).all(), "proximity matrix contains negative infinite values"
    assert (proximity_matrix >= 0).all(), "proximity matrix contains negative values"

    numpy.fill_diagonal(proximity_matrix, numpy.inf)
    distance_matrix = 1.0 / proximity_matrix
    return distance_matrix

def convert_distance_to_proximity_matrix(distance_matrix):
    """
    Function converts the given distance matrix to a proximity matrix and returns it.

    @distance_matrix: a 2d numpy.array(), a matrix that represents the distances between the objects.
    """
    assert type(distance_matrix) == numpy.ndarray, "distance matrix must be numpy.ndarray"
    assert len(distance_matrix.shape) == 2, "distance matrix must have 2 dimensions"
    assert distance_matrix.shape[0] == distance_matrix.shape[1], "distance matrix must be NxN matrix"
    assert (~numpy.isnan(distance_matrix)).all(), "distance matrix contains nan values"
    assert (~numpy.isneginf(distance_matrix)).all(), "distance matrix contains negative infinite values"
    assert (distance_matrix.round(5) >= 0).all(), "distance matrix contains negative values"

    numpy.fill_diagonal(distance_matrix, 0)
    proximity_matrix = 1.0 / distance_matrix
    return proximity_matrix

#######################################################################################################################
################################################ ALGORITHM FUNCTIONS ##################################################
#######################################################################################################################


def return_distance_matrix_dictionary_for_estimators_and_scales(grid, objects_list_normalised, scale_list, estimator_list, choice_parallelization, file_log=None, print_run=True):
    """
    Function calculates the distance matrices for the given scales and estimators.

    @grid: numpy.ndarray(), the grid onto which the objects are interpolated (the x-axis).
    @objects_list_normalised: numpy.ndarray(), the list of the normalized objects.
    @scale_list: a list of the scales to use per estimator, e.g. [[6, 8, 10, 12, 14], [1, 2, 4, 6, 8]]
    @estimator_list: a list of estimators to use, e.g., ["EMD_brute_force", "KL"]
    @file_log: a file object, optional. If a file object is given, the function will save running time of each distance matrix.
    @print_run: boolean, whether to print progress or not, default is True.
    """
    assert type(grid) == numpy.ndarray, "grid must be numpy.ndarray"
    assert type(objects_list_normalised) == numpy.ndarray, "objects_list_normalised must be numpy.ndarray"
    assert ((len(grid.shape) == 1) or (len(grid.shape) == 2)), "objects can be 1 or 2 dimensional"
    assert (~numpy.isnan(grid)).all(), "grid cannot contain nan values"
    assert (~numpy.isinf(grid)).all(), "grid cannot contain infinite values"
    assert (~numpy.isneginf(grid)).all(), "grid cannot contain negative infinite values"
    assert (~numpy.isnan(objects_list_normalised)).all(), "objects_list_normalised cannot contain nan values"
    assert (~numpy.isinf(objects_list_normalised)).all(), "objects_list_normalised cannot contain infinite values"
    assert (~numpy.isneginf(objects_list_normalised)).all(), "objects_list_normalised cannot contain negative infinite values"
    if len(grid.shape) == 1:
        assert (grid.shape[0] == objects_list_normalised.shape[1]), "the grid and the objects must have the same dimensions"
    if len(grid.shape) == 2:
        assert ((grid.shape[0] == objects_list_normalised.shape[1]) and (grid.shape[1] == objects_list_normalised.shape[2])), "the grid and the objects must have the same dimensions"
    assert numpy.fromiter([(isinstance(scale_value, int) or type(scale_value) == numpy.int64) for scale_value in numpy.array(scale_list).flatten()], dtype=bool).all(), "scale values must all be integers"
    assert numpy.fromiter([estimator_value in ['EMD_brute_force', 'energy', 'KL', 'L2'] for estimator_value in estimator_list], dtype=bool).all(), "estimators must be EMD_brute_force, energy, KL or L2"

    distance_matrix_dictionary = {}
    for estimator_index, estimator_name in enumerate(estimator_list):

        scale_list_for_estimator = scale_list[estimator_index]
        for scale_index, scale_value in enumerate(scale_list_for_estimator):

            # printing information and saving it into a log file
            if print_run:
                print("calculating the distance matrices for estimator: %s, scale: %s" % (estimator_name, scale_value))
            if file_log != None:
                file_log.write("calculating the distance matrices for estimator: %s, scale: %s\n" % (estimator_name, scale_value))
                file_log.flush()
            
            start_time = time.time()
            # divide the objects into chunks according to the scale
            N_chunks = scale_value
            grid_splitted, objects_list_splitted = divide_to_chunks(grid, numpy.copy(objects_list_normalised), N_chunks)
            # construct the distance matrix list for this given scale
            distance_matrix_list = []
            for i in range(len(grid_splitted)):
                grid_of_chunk = grid_splitted[i]
                objects_list_of_chunk = objects_list_splitted[i]
                distance_matrix_of_chunk = return_distance_matrix(grid_of_chunk, objects_list_of_chunk, estimator_name, choice_parallelization)
                distance_matrix_list.append(distance_matrix_of_chunk)

            if print_run:    
                print("finished calculating this distance matrix list, it took: %s seconds" % str(time.time() - start_time))
            if file_log != None:
                file_log.write("finished calculating this distance matrix list, it took: %s seconds \n" % str(time.time() - start_time))
                file_log.flush()

            # add the list of matrices to the dictionary
            distance_matrix_dictionary[(estimator_name, scale_value)] = distance_matrix_list
    return distance_matrix_dictionary

def return_weighted_distance_matrix_for_single_estimator_and_scale(distance_matrix_list, to_return_weight_list=True):
    """
    Function takes as an input a list of distance matrices, which correspond to the different chunks at a given scale.
    Function orders the spectra according to each chunk and measures the axis ratio which serves as a weight of each sequence.
    Function then performs a weighted average to return a single distance matrix, according to the axis ratio.

    @distance_matrix_list: a list of distance matrices. Each distance matrix must be numpy.ndarray of NxN.
    @to_return_weight_list: boolean, whether to return the weight (axis ratio) list.
    """
    assert type(distance_matrix_list) == list, "distance_matrix_list must be a list"
    for distance_matrix in distance_matrix_list:
        assert type(distance_matrix) == numpy.ndarray, "distance matrix must be numpy.ndarray"
        assert len(distance_matrix.shape) == 2, "distance matrix must have 2 dimensions"
        assert distance_matrix.shape[0] == distance_matrix.shape[1], "distance matrix must be NxN matrix"
        assert (~numpy.isnan(distance_matrix)).all(), "distance matrix contains nan values"
        assert (~numpy.isinf(distance_matrix)).all(), "distance matrix contains infinite values"
        assert (~numpy.isneginf(distance_matrix)).all(), "distance matrix contains negative infinite values"
        assert (distance_matrix.round(5) >= 0).all(), "distance matrix contains negative values"
        assert (numpy.diagonal(distance_matrix) == 0).all(), "distance matrix must contain zero values in its diagonal"

    weight_list = []
    for chunk_index in range(len(distance_matrix_list)):
        distance_matrix_of_chunk = distance_matrix_list[chunk_index]
        mst, mst_axis_ratio = apply_MST_and_return_MST_and_axis_ratio(distance_matrix_of_chunk, return_axis_ratio=True)
        weight_of_chunk = mst_axis_ratio
        weight_list.append(weight_of_chunk)

    weight_list = numpy.array(weight_list).astype(numpy.float32)

    # now take the weighted average to the list according to the weights you calculated
    weighted_distance_matrix = numpy.average(distance_matrix_list, axis=0, weights=weight_list)
    if to_return_weight_list:
        return weighted_distance_matrix, weight_list
    else:
        return weighted_distance_matrix

def main_of_sequence_parallel(grid, objects_list, estimator_list, scale_list, outpath, to_print_progress=True, \
         to_calculate_distance_matrices=True, to_save_distance_matrices=True, \
         distance_matrices_inpath=None, to_save_axis_ratios=True, return_weighted_products=True, \
         to_average_N_best_estimators=False, number_of_best_estimators=None, \
         to_use_parallelization=False):
    """
    This is the main function of the sequencer. The function takes as an input the grid, objects_list, estimator_list,
    and scale_list and calculates the weighted sequence according to these inputs.
    
    The input to the code:
    @grid: numpy.ndarray(), the grid onto which the objects are interpolated. The grid should consists of float values
           and should not contain nan and infinite values. The data is assumed to be 1 or 2 dimensional.
    @objects_list: numpy.ndarray(), the list of object features to be considered for the sequencing. The objects are
            assumed to be interpolated to a common grid and should not contain nan of infinite values.
            The data is assumed to be 1 or 2 dimensional, therefore the objects list should have 2 or 3 dimensions.
    @estimator_list: list, a list of the estimators to be used for the distance measurement.
            The estimators can be: ['EMD_brute_force', 'energy', 'KL', 'L2']
    @scale_list: list, a list of the scales to use for each estimator. The length of the list is similar to the 
            number of estimators given in the input. The scales must be interger values that correspond to the number 
            of chunks the data is divided to. If the data is one-dimensional, a single chunk value is given for each 
            scale, e.g., scale_list=[[1,2,4], [1,2,4]] if estimator_list=['EMD_brute_force', 'KL']. If the data is 
            two-dimensional, two chunk values are given of each scale, e.g., [[(1,1), (1,2), (2,1)], [(1,1), (1,2), (2,1)]]
            if estimator_list=['EMD_brute_force', 'KL'].
    @outpath: string, a directory path in which the output data and the log file will be saved.
    @to_print_progress: boolean, whether to print in the python shell the progress of the code. Default is True.
    @to_calculate_distance_matrices: boolean, whether to calculate the distance matrices. Default is True, and then the 
            sequencer calculates the distance matrices (which can take a while). If the matrices were already calculated, 
            one should set to_calculate_distance_matrices=False and provide the path of the distance matrices
            using distance_matrices_inpath. 
    @to_save_distance_matrices: boolean, whether to save the distance matrices for future use of the sequencer. The distance
            matrices are saved in a dictionary format which can later be used by the sequencer when setting 
            to_calculate_distance_matrices = False and providing the code with the distance matrices path.
    @distance_matrices_inpath: if to_calculate_distance_matrices == False, function loads the distance matrices from this path.
            The code assumes that the distance matrices are saved in a dictionary format which is similar to the format
            used by the sequencer to save them.
    @to_save_axis_ratios: boolean, whether to save the derived axis ratios for each estimator and scale. Default is True.
            These can be useful to map the important distance measures and scales of the problem.
    @return_weighted_products: boolean, whether to return the weighted distance matrices and their axis ratios.
    @to_average_N_best_estimators: boolean, whether to consider only N best estimators and scales when constructing the final sequence. 
            Default is False, and the sequencer considers the information from all estimators and scales. 
    @number_of_best_estimators: boolean, if to_average_N_best_estimators == True, function constructs the sequence
            while considering only the best number_of_best_estimators (in terms of axis ratio) estimator-scale results.
    @to_use_parallelization: boolean, whether to use parallelization when estimating the distance matrices. The parallelization
            is built for the specific machine we used in our study and will not necessarily work on a different machine.
            Default is False.
    """
    grid = numpy.array(grid)
    objects_list = numpy.array(objects_list)
    N_obj = len(objects_list)

    assert ((len(grid.shape) == 1) or (len(grid.shape) == 2)), "objects can be one- or two-dimensional"
    assert (~numpy.isnan(grid)).all(), "grid cannot contain nan values"
    assert (~numpy.isinf(grid)).all(), "grid cannot contain infinite values"
    assert (~numpy.isneginf(grid)).all(), "grid cannot contain negative infinite values"
    assert ((len(objects_list.shape) == 2) or (len(objects_list.shape) == 3)), "objects can be one- or two-dimensional"
    assert (~numpy.isnan(objects_list)).all(), "objects_list cannot contain nan values"
    assert (~numpy.isinf(objects_list)).all(), "objects_list cannot contain infinite values"
    assert (~numpy.isneginf(objects_list)).all(), "objects_list cannot contain negative infinite values"
    if len(grid.shape) == 1:
        assert (grid.shape[0] == objects_list.shape[1]), "the grid and the objects must have the same dimensions"
    if len(grid.shape) == 2:
        assert ((grid.shape[0] == objects_list.shape[1]) and (grid.shape[1] == objects_list.shape[2])), "the grid and the objects must have the same dimensions"

    assert numpy.fromiter([(isinstance(scale_value, int) or type(scale_value) == numpy.int64) for scale_value in numpy.array(scale_list).flatten()], dtype=bool).all(), "scale values must all be integers"
    assert numpy.fromiter([estimator_value in ['EMD_brute_force', 'energy', 'KL', 'L2'] for estimator_value in estimator_list], dtype=bool).all(), "estimators must be EMD_brute_force, energy, KL or L2"
    assert len(scale_list) == len(estimator_list), "the length of scale_list must equal to the length of estimator_list"
    for scale_value in scale_list:
        scale_shape = numpy.array(scale_value).shape
        assert len(grid.shape) == len(scale_shape), "the shape of scales must be similar to the shape of the data"
        if len(grid.shape) == 1:
            assert scale_shape[0] < grid.shape[0], "the scale must be smaller than the input data"
        if len(grid.shape) == 2:
            assert (scale_shape[0] < grid.shape[0]) and (scale_shape[1] < grid.shape[1]), "the scale must be smaller than the input data"

    assert type(outpath) == str, "output path should be string"
    assert os.path.isdir(outpath), "output path should be a directory"

    if to_calculate_distance_matrices == True: 
        assert distance_matrices_inpath == None, "if to_calculate_distance_matrices=True, distance_matrices_inpath must be None"
    if distance_matrices_inpath != None:
        assert (to_calculate_distance_matrices == False), "if to_calculate_distance_matrices is not None, to_calculate_distance_matrices must be False"
        assert type(distance_matrices_inpath) == str, "distance_matrices_inpath path should be string"
    
    if to_average_N_best_estimators == False: 
        assert number_of_best_estimators == None, "if to_average_N_best_estimators=False, number_of_best_estimators must be None"
    if to_average_N_best_estimators == True:
        assert isinstance(number_of_best_estimators, int), "if to_average_N_best_estimators=True, number_of_best_estimators must be an integer"

    ########################################################################################################
    ######### Parallelization                                                                    ###########
    ######################################################################################################## 
    choice_parallelization = to_use_parallelization
    if choice_parallelization:
        num_cores = multiprocessing.cpu_count()
        if to_print_progress:
            print("Parallelization is ON. Number of cores:",num_cores)
    my_choice_parallelization = choice_parallelization

    ########################################################################################################
    ######### Output files                                                                       ###########
    ######################################################################################################## 
    log_file_path = "%s/log_file.txt" % outpath
    distance_matrices_outpath = "%s/distance_matrices.pkl" % outpath
    axis_ratios_outpath = "%s/axis_ratios.pkl" % outpath
    weighted_distance_matrix_outpath = "%s/weighted_distance_matrix.pkl" % outpath
    sparse_distance_matrix_outpath = "%s/sparse_distance_matrix.pkl" % outpath
    final_products_outpath = "%s/final_products.pkl" % outpath

    file_log = open(log_file_path, "a")
    file_log.write("started the run\n")
    file_log.flush()

    ########################################################################################################
    ######### STEP 1: load or calculate distance matrices for different estimators and scales    ###########
    ########################################################################################################    
    if to_calculate_distance_matrices == False:
        distance_matrix_dictionary = load_distance_matrices_from_path(distance_matrices_inpath, scale_list, estimator_list)

    else:
        objects_list_copy = numpy.copy(objects_list)
        distance_matrix_dictionary = return_distance_matrix_dictionary_for_estimators_and_scales(grid, objects_list_copy, scale_list, estimator_list, choice_parallelization, file_log, to_print_progress)
        # save it if neccessary
        if to_save_distance_matrices == True:
            dump_distance_matrices_to_path(distance_matrices_outpath, distance_matrix_dictionary)
            if to_print_progress:
                print("dumped the distance matrix dictionaries to the file: %s" % distance_matrices_outpath)

    ########################################################################################################
    ######### STEP 2: order the spectra based on the different distance matrices, and measure    ###########
    #########         weights using the MST axis ratios.                                         ###########
    #########         Produce weighted distance matrix per scale and estimator.                  ###########    
    ######################################################################################################## 
    weighted_axis_ratio_dictionary = {}
    # the following lists will be used in STEP 3 for the proximity matrices
    sequences_all_bfs = []
    sequences_all_dfs = []
    weights_all_bfs = []
    weights_all_dfs = []
    MST_list = []
    distance_matrix_all = numpy.zeros((N_obj, N_obj))

    if to_print_progress:
        print("strating to sequence the different scales and estimators")

    for estimator_index, estimator_name in enumerate(estimator_list):
        scale_list_for_estimator = scale_list[estimator_index]

        for scale_index, scale_value in enumerate(scale_list_for_estimator):
            if to_print_progress:
                print("in estimator: %s, scale: %s" % (estimator_name, scale_value))

            distance_matrix_list = distance_matrix_dictionary[(estimator_name, scale_value)]
            weighted_distance_matrix, weight_per_chunk_list = return_weighted_distance_matrix_for_single_estimator_and_scale(distance_matrix_list, to_return_weight_list=True)

            # now obtain sequences from the weighted distance matrix
            ordering_bfs, ordering_dfs, mst_axis_ratio, MST = apply_MST_and_return_BFS_DFS_ordering(weighted_distance_matrix, return_axis_ratio=True, return_MST=True)
            sequences_all_bfs.append(ordering_bfs)
            sequences_all_dfs.append(ordering_dfs)
            MST_list.append(MST)
            distance_matrix_all += (weighted_distance_matrix * mst_axis_ratio)

            # now get the weight estimate, which is the MST axis ratio in this case
            weight_of_sequence_bfs = mst_axis_ratio
            weight_of_sequence_dfs = mst_axis_ratio

            weighted_axis_ratio_dictionary[(estimator_name, scale_value, "chunks")] = weight_per_chunk_list
            weighted_axis_ratio_dictionary[(estimator_name, scale_value, "BFS")] = weight_of_sequence_bfs
            weighted_axis_ratio_dictionary[(estimator_name, scale_value, "DFS")] = weight_of_sequence_dfs
            weights_all_bfs.append(weight_of_sequence_bfs)
            weights_all_dfs.append(weight_of_sequence_dfs)

    if to_save_axis_ratios:
        f_axis_ratios = open(axis_ratios_outpath, "wb")
        pickle.dump(weighted_axis_ratio_dictionary, f_axis_ratios)
        f_axis_ratios.close()
        if to_print_progress:
            print("dumped the axis ratios to the file: %s" % axis_ratios_outpath)

    distance_matrix_all /= numpy.sum(weights_all_bfs)
    numpy.fill_diagonal(distance_matrix_all, 0) 
    f_distance = open(weighted_distance_matrix_outpath, "wb")
    pickle.dump(distance_matrix_all, f_distance)
    f_distance.close()
    if to_print_progress:
        print("dumped the full weighted distance matrix to the file: %s" % weighted_distance_matrix_outpath)

    ########################################################################################################
    ######### STEP 3: use the axis ratios of the weighted distance matrices sequences            ###########
    #########         to build proximity matrices, then convert them to distance matrices,       ###########
    #########         and obtain the final BFS and DFS sequences.                                ###########    
    ######################################################################################################## 

    proximity_matrix_sparse = return_proximity_matrix_populated_by_MSTs_avg_prox(MST_list, weights_all_bfs, to_average_N_best_estimators, number_of_best_estimators)
    distance_matrix_sparse = convert_proximity_to_distance_matrix(proximity_matrix_sparse)
    ordering_bfs, ordering_dfs, mst_axis_ratio = apply_MST_and_return_BFS_DFS_ordering(distance_matrix_sparse, return_axis_ratio=True, return_MST=False)

    ########################################################################################################
    ######### STEP 4: save the final BFS and DFS sequences, their final axis ratio, and          ###########
    #########         the sparse distance matrix that was used to obtain these.                  ###########
    ######################################################################################################## 
    f_distance = open(sparse_distance_matrix_outpath, "wb")
    pickle.dump(distance_matrix_sparse, f_distance)
    f_distance.close()
    if to_print_progress:
        print("dumped the sparse distance matrix to the file: %s" % f_distance)

    final_sequences_dict = {'BFS': ordering_bfs, 'DFS': ordering_dfs}
    f_final_products = open(final_products_outpath, "wb")
    pickle.dump([mst_axis_ratio, final_sequences_dict], f_final_products)
    f_final_products.close()
    if to_print_progress:
        print("dumped the final sequences and axis ratio to the file: %s" % f_final_products)

    # remove the temporary directory and the temporary data if choice_parallelization=True
    if choice_parallelization:
        folder = './joblib_memmap'
        shutil.rmtree(folder)
    
    return mst_axis_ratio, ordering_bfs

