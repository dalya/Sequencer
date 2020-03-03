#######################################################################################################################
####################################################### IMPORTS #######################################################
#######################################################################################################################

import numpy
from scipy.stats import wasserstein_distance
from scipy.stats import energy_distance
from scipy.stats import entropy
import multiprocessing
from joblib import Parallel, delayed

#from emd import emd

#######################################################################################################################
##  Euclidean Distance:                                                                                              ##
##  (1) Can be applied to both 1D (vectors) and 2D (images) without special treatment for the different dimensions.  ##
##  (2) The distance matrix is symmetric, so the calculations can be reduced by a factor of two.                     ##
#######################################################################################################################

def return_L2_mat(objects_list, choice_parallelization):
    """
    The function returns the Euclidean distance matrix for this dataset. 
    The L2 distance is symmetric, and the function uses this to reduce the number of computations.
    """
    n_obj = len(objects_list)
    distance_matrix = numpy.zeros((n_obj, n_obj))

    if choice_parallelization == True:

        master_index_list = []
        for i in range(n_obj):
            for j in range(i+1, n_obj):
                master_index_list.append([i,j])

        ## parallelization
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(l2_simple)(objects_list, i) for i in master_index_list)

        for i_pair in range(len(results)):
            i = master_index_list[i_pair][0]
            j = master_index_list[i_pair][1]
            distance_matrix[i, j] = results[i_pair]
            distance_matrix[j, i] = results[i_pair]

    if choice_parallelization == False:

        for i in range(n_obj):
            for j in range(i+1, n_obj):
                obj_i = objects_list[i]
                obj_j = objects_list[j]
                distance_matrix[i, j] = numpy.sum((obj_i - obj_j)**2)
                distance_matrix[j, i] = numpy.sum((obj_i - obj_j)**2)
    return distance_matrix

def l2_simple(objects_list, index_pair):
    """
    A simple L2 distance measurement to be used with parallelization
    """
    return numpy.sum((objects_list[index_pair[0]] - objects_list[index_pair[1]])**2)


#######################################################################################################################
##  KL Divergence:                                                                                                   ##
##  (1) Can be applied to both 1D (vectors) and 2D (images). For 2D images, the input must be flattened prior to     ##
##      application.                                                                                                 ##
##  (2) The distance matrix is asymmetric.                                                                           ## 
#######################################################################################################################

def return_kl_mat(objects_list, choice_parallelization):
    """
    The function returns the KL-divergence matrix for this dataset. 
    Objects should already be normalised and interpolated to the same grid.
    KL metric is not symmetric, and thus the function performs all the computations.
    """
    n_obj = len(objects_list)
    distance_matrix = numpy.zeros((n_obj, n_obj))

    if choice_parallelization == True:

        master_index_list = []
        for i in range(n_obj):
            for j in range(n_obj):
                master_index_list.append([i, j])

        ## parallelization
        num_cores = multiprocessing.cpu_count()
        if len(objects_list[0].shape) == 1:
            results = Parallel(n_jobs=num_cores)(delayed(kl_simple_1D)(objects_list, i) for i in master_index_list)
        elif len(objects_list[0].shape) == 2:
            results = Parallel(n_jobs=num_cores)(delayed(kl_simple_2D)(objects_list, i) for i in master_index_list)

        for i_pair in range(len(results)):
            i = master_index_list[i_pair][0]
            j = master_index_list[i_pair][1]
            distance_matrix[i, j] = results[i_pair]


    if choice_parallelization == False:

        if len(objects_list[0].shape) == 1:
            for i in range(n_obj):
                for j in range(n_obj):
                    obj_i = objects_list[i]
                    obj_j = objects_list[j]
                    distance_matrix[i, j] = entropy(obj_i, obj_j)

        elif len(objects_list[0].shape) == 2:
            for i in range(n_obj):
                for j in range(n_obj):
                    obj_i_flat = objects_list[i].flatten()
                    obj_j_flat = objects_list[j].flatten()

                    distance_matrix[i, j] = entropy(obj_i_flat, obj_j_flat)

    return distance_matrix

def kl_simple_1D(objects_list, index_pair):
    """
    A simple 1D KL divergence measurement to be used with parallelization
    """
    return entropy(objects_list[index_pair[0]], objects_list[index_pair[1]])

def kl_simple_2D(objects_list, index_pair):
    """
    A simple 2D KL divergence measurement to be used with parallelization
    """
    obj_i = objects_list[index_pair[0]]
    obj_j = objects_list[index_pair[1]]

    obj_i_flat = obj_i.flatten()
    obj_j_small = obj_j.flatten()

    return entropy(obj_i_flat, obj_j_flat)

#######################################################################################################################
##  Earth Mover Distance:                                                                                            ##
##  (1) Can be applied to both 1D (vectors) and 2D (images). For 1D objects, there is a scipy function which is      ##
##      very fast. For 2D objects, we must solve a transportation problem and the computation is very long.          ##
##  (2) The distance matrix is symmetric, so the calculations can be reduced by a factor of two.                     ##
#######################################################################################################################

def return_emd_mat_brute_force(grid, objects_list, choice_parallelization):
    """
    The function returns the EMD distance matrix for this dataset. 
    The objects are assumed to be normalized and interpolated to a common grid.
    The EMD distance is symmetric, and the function uses this to reduce the number of computations.
    """
    n_obj = len(objects_list)
    n_pix = len(grid)
    distance_matrix = numpy.zeros((n_obj, n_obj))

    if choice_parallelization == True:

        master_index_list = []
        for i in range(n_obj):
            for j in range(i+1, n_obj):
                master_index_list.append([i, j])

        ## parallelization
        num_cores = multiprocessing.cpu_count()
        if len(grid.shape) == 1:
            results = Parallel(n_jobs=num_cores)(delayed(emd_simple_1D)(grid, objects_list, i) for i in master_index_list)
        if len(grid.shape) == 2:
            results = Parallel(n_jobs=num_cores)(delayed(emd_simple_2D)(grid, objects_list, i) for i in master_index_list)

        for i_pair in range(len(results)):
            i = master_index_list[i_pair][0]
            j = master_index_list[i_pair][1]
            distance_matrix[i, j] = results[i_pair]
            distance_matrix[j, i] = results[i_pair]


    if choice_parallelization == False:

        if len(grid.shape) == 1:
            
            for i in range(n_obj):
                for j in range(i+1, n_obj):
                    obj_i = objects_list[i]
                    obj_j = objects_list[j]
                    emd_val = wasserstein_distance(grid, grid, u_weights=obj_i, v_weights=obj_j)
                    distance_matrix[i, j] = emd_val
                    distance_matrix[j, i] = emd_val

        if len(grid.shape) == 2:
            for i in range(n_obj):
                for j in range(i+1, n_obj):
                    obj_i = objects_list[i]
                    obj_j = objects_list[j]

                    X = numpy.column_stack(numpy.nonzero(obj_i))
                    Y = numpy.column_stack(numpy.nonzero(obj_j))
                    X_w = obj_i[numpy.nonzero(obj_i)] 
                    Y_w = obj_j[numpy.nonzero(obj_j)]
                    emd_val = emd(X, Y, X_weights=X_w, Y_weights=Y_w)

                    distance_matrix[i, j] = emd_val
                    distance_matrix[j, i] = emd_val

    return distance_matrix

def emd_simple_1D(grid, objects_list, index_pair):
    """
    A simple 1D EMD measurement to be used with parallelization
    """
    return wasserstein_distance(grid, grid, u_weights=objects_list[index_pair[0]], v_weights=objects_list[index_pair[1]])

def emd_simple_2D(grid, objects_list, index_pair):
    """
    A simple 2D EMD measurement to be used with parallelization
    """
    obj_i = objects_list[index_pair[0]]
    obj_j = objects_list[index_pair[1]]

    X = numpy.column_stack(numpy.nonzero(obj_i))
    Y = numpy.column_stack(numpy.nonzero(obj_j))
    X_w = obj_i[numpy.nonzero(obj_i)] 
    Y_w = obj_j[numpy.nonzero(obj_j)]
    return emd(X, Y, X_weights=X_w, Y_weights=Y_w)

#######################################################################################################################
##  Energy Distance:                                                                                                 ##
##  (1) Can only be applied 1D objects.                                                                              ##
##  (2) The distance matrix is symmetric, so the calculations can be reduced by a factor of two.                     ##
#######################################################################################################################

def return_energy_mat(grid, objects_list, choice_parallelization):
    """
    The function returns the energy distance matrix for this dataset. 
    The objects are assumed to be normalized and interpolated to a common grid.
    The energy distance is symmetric, and the function uses this to reduce the number of computations.
    """
    assert len(grid.shape) == 1, "energy distance can only be applied on 1D objects"

    n_obj = len(objects_list)
    n_pix = len(grid)
    distance_matrix = numpy.zeros((n_obj, n_obj))

    if choice_parallelization == True:

        master_index_list = []
        for i in range(n_obj):
            for j in range(i+1, n_obj):
                master_index_list.append([i, j])

        ## parallelization
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(energy_simple_1D)(grid, objects_list, i) for i in master_index_list)

        for i_pair in range(len(results)):
            i = master_index_list[i_pair][0]
            j = master_index_list[i_pair][1]
            distance_matrix[i, j] = results[i_pair]
            distance_matrix[j, i] = results[i_pair]


    if choice_parallelization == False:
            
        for i in range(n_obj):
            for j in range(i+1, n_obj):
                obj_i = objects_list[i]
                obj_j = objects_list[j]
                emd_val = energy_distance(grid, grid, u_weights=obj_i, v_weights=obj_j)
                distance_matrix[i, j] = emd_val
                distance_matrix[j, i] = emd_val

    return distance_matrix

def energy_simple_1D(grid, objects_list, index_pair):
    """
    A simple 1D energy distance measurement to be used with parallelization
    """
    return energy_distance(grid, grid, u_weights=objects_list[index_pair[0]], v_weights=objects_list[index_pair[1]])
