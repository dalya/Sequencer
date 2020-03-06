#######################################################################################################################
####################################################### IMPORTS #######################################################
#######################################################################################################################

import os
import shutil
import pickle

import numpy
import time
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import wasserstein_distance
from scipy.stats import energy_distance
from scipy.stats import entropy
from scipy.interpolate import interp1d
from joblib import Parallel, delayed, dump, load
import multiprocessing

import distance_metrics

#######################################################################################################################
##  Sequencer Class                                                                                                  ##
#######################################################################################################################

class Sequencer(object):
    """An algorithm that detects one-dimensional trends (sequences) in complex datasets. To do so, To do so, it 
    reorders objects within a set to produce the most elongated manifold describing their similarities which are 
    measured in a multi-scale manner and using a collection of metrics. 

    Parameters
    ----------
    :param grid: numpy.ndarray(), the x-axis of the objects in the sample. The grid should consist of float values
                 and should not contain nan or infinite values. Since the data is assumed to be either 1D (vectors) 
                 or 2D (matrices), the grid is assumed to be 1D or 2D as well.

    :param objects_list: numpy.ndarray(), the list of the objects to sequence. The objects are
                         assumed to be interpolated to a common grid and should not contain nan of infinite values.
                         The data is assumed to be 1 or 2 dimensional, therefore the objects list should have 2 or 
                         3 dimensions.

    :param estimator_list: list of strings (default=['EMD', 'energy', 'KL', 'L2']) , a list of estimators to be used f
                           or the distance assignment. The current available estimators are: 'EMD', 'energy', 'KL', and 'L2'. 

    :param scale_list: list of integers or None (default=None). A list of the scales to use for each estimator. The 
                       length of the list is similar to the number of estimators given in the input. The scales must 
                       be interger values that correspond to the number of parts the data is divided to. If the data 
                       is one-dimensional, a single chunk value is given for each scale, e.g., scale_list=[[1,2,4], [1,2,4]] 
                       if estimator_list=['EMD', 'KL']. This means that the sequencer will divide each 
                       object into 1 parts (full object), 2 parts (splitting each object into 2 parts), and 4 parts.
                       If the data is two-dimensional, two chunk values are given of each scale, the first describes
                       the horizontal direction and the seconds describes the vertical direction. For example, we will
                       set: scale_list= [[(1,1), (1,2), (2,1)], [(1,1), (1,2), (2,1)]] if 
                       estimator_list=['EMD', 'KL']. The scales can be different for different estimators.

                       if scale_list=None, then default list of scales is calculated using different powers of 2, such
                       that the minimal length of a part is 20 pixels. For example, if the length of the objects is 100,
                       then the default scales will be: 1, 2, 4. If the length of the objects is 1000, then the default
                       scales will be: 1, 2, 4, 8, 16, 32. For 2D objects, the set of default scales is [1, 1].
    """
    def __init__(self, grid, objects_list, estimator_list, scale_list=None):
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

        if scale_list != None:
            assert numpy.fromiter([(isinstance(scale_value, int) or type(scale_value) == numpy.int64) for scale_value in numpy.array(scale_list).flatten()], dtype=bool).all(), "scale values must all be integers"
        assert numpy.fromiter([estimator_value in ['EMD', 'energy', 'KL', 'L2'] for estimator_value in estimator_list], dtype=bool).all(), "estimators must be EMD, energy, KL or L2"
        if scale_list != None:
            assert len(scale_list) == len(estimator_list), "the length of scale_list must equal to the length of estimator_list"
            for scale_value in scale_list:
                scale_shape = numpy.array(scale_value).shape
                assert len(grid.shape) == len(scale_shape), "the shape of scales must be similar to the shape of the data"
                if len(grid.shape) == 1:
                    assert scale_shape[0] < grid.shape[0], "the scale must be smaller than the input data"
                if len(grid.shape) == 2:
                    assert (scale_shape[0] < grid.shape[0]) and (scale_shape[1] < grid.shape[1]), "the scale must be smaller than the input data"

        self.grid = grid
        self.objects_list = objects_list
        self.estimator_list = estimator_list
        if scale_list != None:
            self.scale_list = scale_list
        else:
            if len(grid.shape) == 1:
                length_of_object = len(objects_list[0])
                if length_of_object > 20:
                    maximal_scale_size = length_of_object / 20.
                    scale_list_for_estimator = list(2**numpy.arange(0, numpy.log2(maximal_scale_size)))
                else:
                    scale_list_for_estimator = [1]
                scale_list = [scale_list_for_estimator] * len(self.estimator_list)

                self.scale_list = scale_list

            else: # len(grid.shape) == 2:
                scale_list_for_estimator = [(1,1)]
                scale_list = [scale_list_for_estimator] * len(self.estimator_list)
                self.scale_list = scale_list

        # set to None the parameters that are calculated during the execute function
        self.weighted_axis_ratio_and_sequence_dictionary = None


    def execute(self, outpath, to_print_progress=True, to_calculate_distance_matrices=True, to_save_distance_matrices=True, \
        distance_matrices_inpath=None, to_save_axis_ratios=True, to_average_N_best_estimators=False, number_of_best_estimators=None, \
        to_use_parallelization=False):
        """Main function of the sequencer that applies the algorithm to the data, and returns the best sequence and its axis ratio.
        (*) The function can save many intermediate products, such as the distance matrices for each estimator and scale. The user is 
        encoraged to save these products, since they can be used later to reduce dramatically the computation time. 
        (*) The function also allows the user to perform the majority vote using the N best estimators, instead of using all of 
        them.

        Parameters
        ----------
        :param outpath: string, the path of a directory to which the function will save intermediate products and the log file.

        :param to_print_progress: boolean (default=True), whether to print the progress of the code. 

        :param to_calculate_distance_matrices: boolean (default=True), whether to calculate the distance matrices per estimator
            and scale. If True, the distance matrices per estimator and scale are eatimated. If the distance matrices were already
            calculated in a previous run of the function, the user is encoraged to set to_calculate_distance_matrices=False and 
            provide a path where the matrices are available using distance_matrices_inpath.

        :param to_save_distance_matrices: boolean (default=True), whether to save the estimated distance matrices or not. The user 
            strongly encoraged to save the matrices in order to reduce the computation time of future runs.

        :param distance_matrices_inpath: string (default=None), the input path of the distance matrices. If the distance matrices
            were already estimated in a previous run of the function, the user can avoid the re-computation of the distance matrices
            and load them from the given path. This can be done by setting to_calculate_distance_matrices=False and providing the 
            input path of the distance matrices.

        :param to_save_axis_ratios: boolean (default=True), whether to save the derived axis ratios for each estimator and scale. 
            For each scale, the function will also save the derived axis ratios of each part (chunk) into which the objects are 
            split into. These values can be useful to map the important metrics and scales of the problem.

        :param to_average_N_best_estimators: boolean (default=False), whether to consider only N best metrics + scales when 
            constructing the final sequence. If to_average_N_best_estimators=True, the function will perform a majority vote
            only considering the N estimators (metrics + scales) with the highest axis ratios.

        :param number_of_best_estimators: integer (default=None), the number of estimators to consider in the majority vote. 
            If to_average_N_best_estimators=True, then the user must provide an integer number. 

        :param to_use_parallelization: boolean (default=False), whether to use parallelization when estimating the distance matrices. The parallelization

        Returns
        -------
        :param final_mst_axis_ratio: float, the final axis ratio of the detected sequence. 
            This is obtained after populating the separate sequences for the different metrics+scales, averaged according to 
            their respective axis ratios. See the paper for additional details.
        :param final_sequence: numpy.ndarray of integers, the final detected sequence. 
        """
        N_obj = len(self.objects_list)

        assert ((len(self.grid.shape) == 1) or (len(self.grid.shape) == 2)), "objects can be one- or two-dimensional"
        assert (~numpy.isnan(self.grid)).all(), "grid cannot contain nan values"
        assert (~numpy.isinf(self.grid)).all(), "grid cannot contain infinite values"
        assert (~numpy.isneginf(self.grid)).all(), "grid cannot contain negative infinite values"
        assert ((len(self.objects_list.shape) == 2) or (len(self.objects_list.shape) == 3)), "objects can be one- or two-dimensional"
        assert (~numpy.isnan(self.objects_list)).all(), "objects_list cannot contain nan values"
        assert (~numpy.isinf(self.objects_list)).all(), "objects_list cannot contain infinite values"
        assert (~numpy.isneginf(self.objects_list)).all(), "objects_list cannot contain negative infinite values"
        if len(self.grid.shape) == 1:
            assert (self.grid.shape[0] == self.objects_list.shape[1]), "the grid and the objects must have the same dimensions"
        if len(self.grid.shape) == 2:
            assert ((self.grid.shape[0] == self.objects_list.shape[1]) and (self.grid.shape[1] == self.objects_list.shape[2])), "the grid and the objects must have the same dimensions"

        assert numpy.fromiter([(isinstance(scale_value, int) or type(scale_value) == numpy.int64) for scale_value in numpy.array(self.scale_list).flatten()], dtype=bool).all(), "scale values must all be integers"
        assert numpy.fromiter([estimator_value in ['EMD', 'energy', 'KL', 'L2'] for estimator_value in self.estimator_list], dtype=bool).all(), "estimators must be EMD, energy, KL or L2"
        assert len(self.scale_list) == len(self.estimator_list), "the length of scale_list must equal to the length of estimator_list"
        for scale_value in self.scale_list:
            scale_shape = numpy.array(scale_value).shape
            assert len(self.grid.shape) == len(scale_shape), "the shape of scales must be similar to the shape of the data"
            if len(self.grid.shape) == 1:
                assert scale_shape[0] < self.grid.shape[0], "the scale must be smaller than the input data"
            if len(self.grid.shape) == 2:
                assert (scale_shape[0] < self.grid.shape[0]) and (scale_shape[1] < self.grid.shape[1]), "the scale must be smaller than the input data"

        assert type(outpath) == str, "outpath should be string"
        assert os.path.isdir(outpath), "outpath should be a directory"

        if to_calculate_distance_matrices == True: 
            assert distance_matrices_inpath == None, "if to_calculate_distance_matrices=True, distance_matrices_inpath must be None"
        if distance_matrices_inpath != None:
            assert (to_calculate_distance_matrices == False), "if distance_matrices_inpath is not None, to_calculate_distance_matrices must be False"
            assert type(distance_matrices_inpath) == str, "distance_matrices_inpath path should be string"
        
        if to_average_N_best_estimators == False: 
            assert number_of_best_estimators == None, "if to_average_N_best_estimators=False, number_of_best_estimators must be None"
        if to_average_N_best_estimators == True:
            assert isinstance(number_of_best_estimators, int), "if to_average_N_best_estimators=True, number_of_best_estimators must be an integer"

        self.outpath = outpath
        self.to_print_progress = to_print_progress
        self.to_calculate_distance_matrices = to_calculate_distance_matrices
        self.to_save_distance_matrices = to_save_distance_matrices
        self.distance_matrices_inpath = distance_matrices_inpath
        self.to_save_axis_ratios = to_save_axis_ratios
        self.to_average_N_best_estimators = to_average_N_best_estimators
        self.number_of_best_estimators = number_of_best_estimators
        self.to_use_parallelization = to_use_parallelization

        ########################################################################################################
        ######### Parallelization                                                                    ###########
        ######################################################################################################## 
        if self.to_use_parallelization:
            num_cores = multiprocessing.cpu_count()
            if self.to_print_progress:
                print("Parallelization is ON. Number of cores:", num_cores)

        ########################################################################################################
        ######### Output files                                                                       ###########
        ######################################################################################################## 
        self.log_file_outpath = "%s/log_file.txt" % self.outpath
        self.distance_matrices_outpath = "%s/distance_matrices.pkl" % self.outpath
        self.axis_ratios_outpath = "%s/axis_ratios.pkl" % self.outpath
        self.weighted_distance_matrix_outpath = "%s/weighted_distance_matrix.pkl" % self.outpath
        self.sparse_distance_matrix_outpath = "%s/sparse_distance_matrix.pkl" % self.outpath
        self.final_products_outpath = "%s/final_products.pkl" % self.outpath

        self.file_log = open(self.log_file_outpath, "a")
        self.file_log.write("started the run\n")
        self.file_log.flush()

        ########################################################################################################
        ######### STEP 1: load or calculate distance matrices for different estimators and scales    ###########
        ########################################################################################################    
        if self.to_calculate_distance_matrices == False:
            distance_matrix_dictionary = self._load_distance_matrices_from_path()

        else:
            distance_matrix_dictionary = self._return_distance_matrix_dictionary_for_estimators_and_scales()
            # save it if neccessary
            if self.to_save_distance_matrices == True:
                self._dump_distance_matrices_to_path(distance_matrix_dictionary)
                if self.to_print_progress:
                    print("dumped the distance matrix dictionaries to the file: %s" % self.distance_matrices_outpath)

        ########################################################################################################
        ######### STEP 2: order the spectra based on the different distance matrices, and measure    ###########
        #########         weights using the MST axis ratios.                                         ###########
        #########         Produce weighted distance matrix per scale and estimator.                  ###########    
        ######################################################################################################## 
        self.weighted_axis_ratio_and_sequence_dictionary = {}
        # the following lists will be used in STEP 3 for the proximity matrices
        MST_list = []
        weight_list = []
        distance_matrix_all = numpy.zeros((N_obj, N_obj))

        if self.to_print_progress:
            print("strating to sequence the different scales and estimators")

        for estimator_index, estimator_name in enumerate(self.estimator_list):
            scale_list_for_estimator = self.scale_list[estimator_index]
            for scale_index, scale_value in enumerate(scale_list_for_estimator):
                if self.to_print_progress:
                    print("in estimator: %s, scale: %s" % (estimator_name, scale_value))

                distance_matrix_list = distance_matrix_dictionary[(estimator_name, scale_value)]
                weighted_distance_matrix, ordering_per_chunk_list, axis_ratio_per_chunk_list = self._return_weighted_distance_matrix_for_single_estimator_and_scale(distance_matrix_list, to_return_axis_ratio_list=True)
                # now obtain sequences from the weighted distance matrix
                ordering_bfs, ordering_dfs, mst_axis_ratio, MST = self._apply_MST_and_return_BFS_DFS_ordering(weighted_distance_matrix, return_axis_ratio=True, return_MST=True)

                MST_list.append(MST)
                weight_list.append(mst_axis_ratio)
                distance_matrix_all += (weighted_distance_matrix * mst_axis_ratio)
                # add the sequences and their axis ratios into a dictionary
                self.weighted_axis_ratio_and_sequence_dictionary[(estimator_name, scale_value, "chunks")] = (axis_ratio_per_chunk_list, ordering_per_chunk_list)
                self.weighted_axis_ratio_and_sequence_dictionary[(estimator_name, scale_value, "weighted")] = (mst_axis_ratio, ordering_bfs)


        if self.to_save_axis_ratios:
            f_axis_ratios = open(self.axis_ratios_outpath, "wb")
            pickle.dump(self.weighted_axis_ratio_and_sequence_dictionary, f_axis_ratios)
            f_axis_ratios.close()
            if self.to_print_progress:
                print("dumped the axis ratios to the file: %s" % self.axis_ratios_outpath)

        distance_matrix_all /= numpy.sum(weight_list)
        numpy.fill_diagonal(distance_matrix_all, 0) 
        f_distance = open(self.weighted_distance_matrix_outpath, "wb")
        pickle.dump(distance_matrix_all, f_distance)
        f_distance.close()
        if self.to_print_progress:
            print("dumped the full weighted distance matrix to the file: %s" % self.weighted_distance_matrix_outpath)

        ########################################################################################################
        ######### STEP 3: use the axis ratios of the weighted distance matrices sequences            ###########
        #########         to build proximity matrices, then convert them to distance matrices,       ###########
        #########         and obtain the final BFS and DFS sequences.                                ###########    
        ######################################################################################################## 

        proximity_matrix_sparse = self._return_proximity_matrix_populated_by_MSTs_avg_prox(MST_list, weight_list)
        distance_matrix_sparse = self._convert_proximity_to_distance_matrix(proximity_matrix_sparse)
        ordering_bfs, ordering_dfs, mst_axis_ratio = self._apply_MST_and_return_BFS_DFS_ordering(distance_matrix_sparse, return_axis_ratio=True, return_MST=False)

        self.final_mst_axis_ratio = mst_axis_ratio
        self.final_sequence = ordering_bfs
        ########################################################################################################
        ######### STEP 4: save the final BFS and DFS sequences, their final axis ratio, and          ###########
        #########         the sparse distance matrix that was used to obtain these.                  ###########
        ######################################################################################################## 
        f_distance = open(self.sparse_distance_matrix_outpath, "wb")
        pickle.dump(distance_matrix_sparse, f_distance)
        f_distance.close()
        if self.to_print_progress:
            print("dumped the sparse distance matrix to the file: %s" % f_distance)

        final_sequences_dict = {'BFS': ordering_bfs, 'DFS': ordering_dfs}
        f_final_products = open(self.final_products_outpath, "wb")
        pickle.dump([mst_axis_ratio, final_sequences_dict], f_final_products)
        f_final_products.close()
        if self.to_print_progress:
            print("dumped the final sequences and axis ratio to the file: %s" % f_final_products)

        # remove the temporary directory and the temporary data if choice_parallelization=True
        if self.to_use_parallelization:
            folder = './joblib_memmap'
            shutil.rmtree(folder)
        
        return self.final_mst_axis_ratio, self.final_sequence


    def return_axis_ratios_and_sequences_per_chunk(self, estimator_name, scale):
        """Function returns the intermediate axis ratios and sequences obtained during the calculation of the final sequence.
        For each distance metric and scale, the sequencer divided each object into different chunks (parts), and estimated 
        its corresponding sequence and axis ratio. This funciton returns a list of these axis ratios and sequences.

        Parameters
        ----------
        :param estimator_name: string, the distance metric for which to return the list of axis ratios and sequences.

        :param scale: integer, the scale for which toe return the list of axis ratios and sequences.


        Returns
        -------
        :param axis_ratio_list: a list of float values, the list of axis ratios obtained for each of the chunks for the given 
            distance metric and scale.

        :param sequence_list: a list of lists, the list of sequences calculated for each of the chunks for the given 
            distance metric and scale.
        """
        assert (self.weighted_axis_ratio_and_sequence_dictionary != None), "the axis ratio and sequence dictionary is empty. Are you sure you executed the sequencer using Sequencer.execute first?"
        assert (estimator_name in self.estimator_list), "the required estimator is not included in the esitmator list"
        for i, estimator_value in enumerate(self.estimator_list):
            if estimator_value == estimator_name:
                scale_list_for_estimator = self.scale_list[i]
                assert (scale in scale_list_for_estimator), "the required scale is not included in the scale list for the given estimator"

        axis_ratio_list, sequence_list = self.weighted_axis_ratio_and_sequence_dictionary[(estimator_name, scale, "chunks")]

        return axis_ratio_list, sequence_list


    def return_axis_ratio_of_weighted_products(self, estimator_name, scale):
        """Function returns the intermediate axis ratios obtained in the second stage of the code. For each distance metric and
        scale, the sequencer estimated the weighted distance matrix, and used it to calculate a sequence and an axis ratio. 

        Parameters
        ----------
        :param estimator_name: string, the distance metric for which to return the list of axis ratios and sequences.

        :param scale: integer, the scale for which toe return the list of axis ratios and sequences.


        Returns
        -------
        :param axis_ratio: a float, the axis ratio that corresponds to the given metric and scale.
        """
        assert (self.weighted_axis_ratio_and_sequence_dictionary != None), "the axis ratio and sequence dictionary is empty. Are you sure you executed the sequencer using Sequencer.execute first?"
        assert (estimator_name in self.estimator_list), "the required estimator is not included in the esitmator list"
        for i, estimator_value in enumerate(self.estimator_list):
            if estimator_value == estimator_name:
                scale_list_for_estimator = self.scale_list[i]
                assert (scale in scale_list_for_estimator), "the required scale is not included in the scale list for the given estimator"   

        axis_ratio, sequence = self.weighted_axis_ratio_and_sequence_dictionary[(estimator_name, scale, "weighted")]

        return axis_ratio


    def return_sequence_of_weighted_products(self, estimator_name, scale):
        """Function returns the intermediate sequence obtained in the second stage of the code. For each distance metric and
        scale, the sequencer estimated the weighted distance matrix, and used it to calculate a sequence and an axis ratio. 

        Parameters
        ----------
        :param estimator_name: string, the distance metric for which to return the list of axis ratios and sequences.

        :param scale: integer, the scale for which toe return the list of axis ratios and sequences.

        Returns
        -------
        :param sequence: a list of integers, the sequence that corresponds to the given metric and scale.
        """
        assert (self.weighted_axis_ratio_and_sequence_dictionary != None), "the axis ratio and sequence dictionary is empty. Are you sure you executed the sequencer using Sequencer.execute first?"
        assert (estimator_name in self.estimator_list), "the required estimator is not included in the esitmator list"
        for i, estimator_value in enumerate(self.estimator_list):
            if estimator_value == estimator_name:
                scale_list_for_estimator = self.scale_list[i]
                assert (scale in scale_list_for_estimator), "the required scale is not included in the scale list for the given estimator"   

        axis_ratio, sequence = self.weighted_axis_ratio_and_sequence_dictionary[(estimator_name, scale, "weighted")]

        return sequence


    def return_axis_ratio_of_weighted_products_all_metrics_and_scales(self):
        """Function returns the intermediate axis ratios obtained in the second stage of the code. For each distance metric and
        scale, the sequencer estimated the weighted distance matrix, and used it to calculate a sequence and an axis ratio. 
        (*) This function returns a list of axis ratios that corresponds to all the different metrics and scales. 
        (*) If the user is interested in a particular metric and scale, then one can use the function: return_axis_ratio_of_weighted_products.

        Returns
        -------
        :param estimator_list: a list of strings, the distance metrics for which the axis ratios were calculated.
        :param scale_list: a list of integers, the scales for which the axis ratios were calculated.
        :param axis_ratio_list: a list of floats, the axis ratios that corresponds to every metric and scale.
        """
        assert (self.weighted_axis_ratio_and_sequence_dictionary != None), "the axis ratio and sequence dictionary is empty. Are you sure you executed the sequencer using Sequencer.execute first?"

        estimator_list = []
        scale_list = []
        axis_ratio_list = []
        for estimator_index, estimator_value in enumerate(self.estimator_list):
            scale_list_for_estimator = self.scale_list[estimator_index]
            for scale_index, scale_value in enumerate(scale_list_for_estimator):
                axis_ratio, sequence = self.weighted_axis_ratio_and_sequence_dictionary[(estimator_value, scale_value, "weighted")]

                estimator_list.append(estimator_value)
                scale_list.append(scale_value)
                axis_ratio_list.append(axis_ratio)

        return estimator_list, scale_list, axis_ratio_list


    def return_sequence_of_weighted_products_all_metrics_and_scales(self):
        """Function returns the intermediate sequences obtained in the second stage of the code. For each distance metric and
        scale, the sequencer estimated the weighted distance matrix, and used it to calculate a sequence and an axis ratio. 
        (*) This function returns a list of sequences that corresponds to all the different metrics and scales. 
        (*) If the user is interested in a particular metric and scale, then one can use the function: return_sequence_of_weighted_products.

        Returns
        -------
        :param estimator_list: a list of strings, the distance metrics for which the axis ratios were calculated.
        :param scale_list: a list of integers, the scales for which the axis ratios were calculated.
        :param sequence_list: a list of lists, the sequences that corresponds to every metric and scale.
        """
        assert (self.weighted_axis_ratio_and_sequence_dictionary != None), "the axis ratio and sequence dictionary is empty. Are you sure you executed the sequencer using Sequencer.execute first?"

        estimator_list = []
        scale_list = []
        sequence_list = []
        for estimator_index, estimator_value in enumerate(self.estimator_list):
            scale_list_for_estimator = self.scale_list[estimator_index]
            for scale_index, scale_value in enumerate(scale_list_for_estimator):
                axis_ratio, sequence = self.weighted_axis_ratio_and_sequence_dictionary[(estimator_value, scale_value, "weighted")]

                estimator_list.append(estimator_value)
                scale_list.append(scale_value)
                sequence_list.append(sequence)

        return estimator_list, scale_list, sequence_list

    #######################################################################################################################
    ################################################ PRIVATE FUNCTIONS ####################################################
    #######################################################################################################################

    
    ###################################################### DATA I/O #######################################################
    def _load_distance_matrices_from_path(self):
        """Funciton loads the distance matrices into memory from the distance matrix input file.

        This is an internal function that is used only in a case where the distance matrices per metric and scale were already
        computed during a previous run, and were saved. In such a case, the user can choose to load the precomputed matrices
        instead of calculating them again. This can save a lot of execution time.

        Returns
        -------
        :param distance_matrix_dictionary: a dictionary where each key is a tuple (estimator_name, scale_value), and the value 
            is a list of distance matrices computed for each chunk of the data. 
        """
        input_file = open(path, "rb")
        distance_matrix_dictionary_saved = pickle.load(self.distance_matrices_inpath)
        input_file.close()

        distance_matrix_dictionary = {}
        for estimator_index, estimator_value in enumerate(self.estimator_list):

            scale_list_for_estimator = self.scale_list[estimator_index]
            for scale_value in scale_list_for_estimator:
                assert ((estimator_value, scale_value) in list(distance_matrix_dictionary_saved.keys())), "the list of saved distance matrices does not include the required metrics and scales by Sequencer.execute"
                distance_matrix_dictionary[(estimator_value, scale_value)] = distance_matrix_dictionary_saved[(estimator_value, scale_value)]

        return distance_matrix_dictionary

    def _dump_distance_matrices_to_path(self, distance_matrix_dictionary):
        """Function saves the provided distance matrix dictionary into a file.
        """
        output_file = open(self.distance_matrices_outpath, "wb")
        pickle.dump(distance_matrix_dictionary, output_file)
        output_file.close()


    ################################################# Distance measures ###################################################
    def _return_distance_matrix(self, grid, objects_list, estimator):
        """Function estimates the distance matrix of the given objects using the given estimator.

        Parameters
        -------
        :param grid: numpy.ndarray(), the x-axis of the objects in the sample
        :param objects_list: a list of numpy.ndarray(), the objects in the sample
        :param estimator: string, the name of the distance metric to use for the estimation of distance

        Returns
        -------
        :param distance_matrix: numpy.ndarray(), the distance matrix 
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
        assert estimator in ['EMD', 'energy', 'KL', 'L2'], "the distance estimator must be: EMD, energy, KL, or L2"
        if estimator == "EMD":
            assert (objects_list >= 0).all(), "the EMD distance can only be applied to non-negative values"
        if estimator == "energy":
            assert (objects_list >= 0).all(), "the energy distance can only be applied to non-negative values"
        if estimator == "KL":
            assert (objects_list > 0).all(), "the KL distance can only be applied to positive values"

        # if the distance claculation should be carried out in parallel, save the data that will be used
        if self.to_use_parallelization:
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

        if estimator == "EMD":
            distance_matrix = distance_metrics.return_emd_mat_brute_force(grid, objects_list, self.to_use_parallelization)

        if estimator == "energy":
            distance_matrix = distance_metrics.return_energy_mat(grid, objects_list, self.to_use_parallelization)

        if estimator == "KL":
            distance_matrix = distance_metrics.return_kl_mat(objects_list, self.to_use_parallelization)

        if estimator == "L2":
            distance_matrix = distance_metrics.return_L2_mat(objects_list, self.to_use_parallelization)

        return distance_matrix

    ################################################## Scale functions ####################################################
    def _normalise_objects(self, objects_list):
        """Function normalizes each of the objects in the list such that the sum of its elements will be one.

        Parameters
        -------
        :param objects_list: a list of numpy.ndarray(), the objects in the sample

        Returns
        -------
        :param objects_list: a list of numpy.ndarray(), the normalized objects
        """
        assert ((len(objects_list.shape) == 2) or (len(objects_list.shape) == 3)), "objects can be either 1D or 2D"

        if len(objects_list.shape) == 2:
            sum_vector = numpy.sum(objects_list, axis=1)
            objects_list_normalised = objects_list / sum_vector[:, numpy.newaxis]

        if len(objects_list.shape) == 3:
            sum_vector = numpy.sum(objects_list, axis=(1,2))
            objects_list_normalised = objects_list / sum_vector[:, numpy.newaxis, numpy.newaxis]

        return objects_list_normalised

    def _divide_to_chunks_1D(self, grid, objects_list, N_chunks):
        """Function divides the data into chunks according to N_chunks, and then normalizes each chunk to have a sum of one.
        Function also splits the grid array into the same chunks. Function assumes that the grid is 1D and the object list 
        is 2D.

        Parameters
        -------
        :param grid: numpy.ndarray(), the x-axis of the objects
        :param objects_list: a list of numpy.ndarray(), the objects in the sample
        :param N_chunks: integer, the number of chunks to split each object into

        Returns
        -------
        :param grid_split: a list of numpy.ndarray(), a list containing the different parts of the split grid
        :param objects_list_split_normalised: a list of numpy.ndarray(), a list of length N_chunks consisting of the 
            objects_list for each chunk, after normalization
        """
        grid_split = numpy.array_split(grid, N_chunks)
        objects_list_split = numpy.array_split(objects_list, N_chunks, axis=-1)
        objects_list_split_normalised = []
        for objects_list_chunk in objects_list_split:
            sum_vec = numpy.sum(objects_list_chunk, axis=-1)
            object_list_chunk_norm = objects_list_chunk / sum_vec[:, numpy.newaxis]
            objects_list_split_normalised.append(object_list_chunk_norm)
                
        return grid_split, objects_list_split_normalised

    def _divide_to_chunks_2D(self, grid, objects_list, N_chunks):
        """Function divides the data into chunks according to N_chunks, and then normalizes each chunk to have a sum of one.
        Function also splits the grid array into the same chunks. Function assumes that the grid is 2D and the object list 
        is 3D.

        Parameters
        -------
        :param grid: numpy.ndarray(), the x-axis of the objects
        :param objects_list: a list of numpy.ndarray(), the objects in the sample
        :param N_chunks: integer, the number of chunks to split each object into

        Returns
        -------
        :param grid_split: a list of numpy.ndarray(), a list containing the different parts of the split grid
        :param objects_list_split_normalised: a list of numpy.ndarray(), a list of length N_chunks consisting of the 
            objects_list for each chunk, after normalization
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

    def _divide_to_chunks(self, grid, objects_list, N_chunks):
        """The main function that divides the data into chunks. It splits the grid and objects_list into chunks
        according to N_chunks. The function does not assume that N_chunks can divide the data into equaly-sized chunks, 
        but instead returns chunks of roughly similar size.

        Parameters
        -------
        :param grid: numpy.ndarray(), the x-axis of the objects
        :param objects_list: a list of numpy.ndarray(), the objects in the sample
        :param N_chunks: integer, the number of chunks to split each object into

        Returns
        -------
        :param grid_split: a list of numpy.ndarray(), a list containing the different parts of the split grid
        :param objects_list_split_normalised: a list of numpy.ndarray(), a list of length N_chunks consisting of the 
            objects_list for each chunk, after normalization
        """
        assert ((len(grid.shape) == 1) or (len(grid.shape) == 2)), "objects can be either 1D or 2D"

        if len(grid.shape) == 1:
            return self._divide_to_chunks_1D(grid, objects_list, N_chunks)
        if len(grid.shape) == 2:
            return self._divide_to_chunks_2D(grid, objects_list, N_chunks)


    ################################################## GRAPH FUNCTIONS ####################################################
    def _return_MST_axis_ratio(self, distance_arr):
        """Function estimates the axis ratio of the MST that is described by the given distance array. The input distance 
        array represents the distances of each node in the graph from the root of the graph (the starting point). 
        Funciton calculates the axis ratio by dividing the half-length of the tree by the half-width.
        The half-width is calculated as the average width in every depth level, and the half-length is calculated as the
        average distance from the root.

        Parameters
        -------
        :param distance_arr: list, a list that described the distance of each node from the root of the tree

        Returns
        -------
        :param mst_axis_ratio: float, the axis ratio of the MST
        """
        graph_half_length = numpy.average(distance_arr) 
        g_unique, counts = numpy.unique(distance_arr, return_counts=True)
        graph_half_width = numpy.average(counts) / 2.
        mst_axis_ratio = float(graph_half_length) / float(graph_half_width) + 1 

        return mst_axis_ratio

    def _return_start_index_from_MST(self, graph):
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

    def _apply_MST_and_return_MST_and_axis_ratio(self, distance_matrix, return_axis_ratio=True):
        """Function converts the distance matrix into a fully-conncted graph and calculates its Minimum Spanning Tree (MST).
        Function has an option to return the axis ratio of the resulting MST. 

        Parameters
        -------
        :param distance_matrix: numpy.ndarray(), the distance matrix that will be converted into an MST.
        :param return_axis_ratio: boolean (default=True), whether to return the axis ratio of the resulting MST.

        Returns
        -------
        :param G: networkx.classes.graph.Graph(), the graph that represents the resulting MST.
        :param mst_axis_ratio (optional): float, the axis ratio of the resulting MST.
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
            start_index = self._return_start_index_from_MST(G)
            distance_dict = nx.shortest_path_length(G, start_index)
            distance_arr = numpy.fromiter(distance_dict.values(), dtype=int)
            mst_axis_ratio = self._return_MST_axis_ratio(distance_arr)
            return G, mst_axis_ratio
        else:
            return G

    def _apply_MST_and_return_BFS_DFS_ordering(self, distance_matrix, start_index=None, return_axis_ratio=True, return_MST=True):
        """Function converts the distance matrix into a fully-connected graph and calculates its minimum spanning tree (MST).
        The function also returns two walks within the tree: BFS and DFS. The function also has an option to return the axis 
        ratio of the resulting MST.

        Parameters
        -------
        :param distance_matrix: numpy.ndarray(), the distance matrix that will be converted into an MST.
        :param start_index: integer (default=None), the index in the matrix from which to start the BFS/DFS walk within the MST. 
            If start_index==None, the function estimates the staring point using the closeness centrality measure.
        :param return_axis_ratio: boolean (default=True), whether to return the axis ratio of the resulting MST.
        :param return MST: boolean (default=True), whether to return the resulting MST.

        Returns
        -------
        :param ordering_bfs: a list of integers, a list representing the indices of the nodes according to a BFS walk in the MST
        :param ordering_dfs: a list of integers, a list representing the indices of the nodes according to a DFS walk in the MST
        :param mst_axis_ratio (optional): float, the axis ratio of the resulting MST.
        :param G (optional): networkx.classes.graph.Graph(), the graph that represents the resulting MST.
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
            start_index = self._return_start_index_from_MST(G)
        # DFS walk
        ordering_dfs = numpy.fromiter(nx.dfs_preorder_nodes(G, start_index), dtype=int)
        # BFS walk
        distance_dict = nx.shortest_path_length(G, start_index)
        distance_inds = numpy.fromiter(distance_dict.keys(), dtype=int)
        distance_arr = numpy.fromiter(distance_dict.values(), dtype=int)
        ordering_bfs = distance_inds[numpy.argsort(distance_arr)]

        if return_axis_ratio and return_MST:
            mst_axis_ratio = self._return_MST_axis_ratio(distance_arr)
            return ordering_bfs, ordering_dfs, mst_axis_ratio, G
        elif return_axis_ratio:
            mst_axis_ratio = self._return_MST_axis_ratio(distance_arr)
            return ordering_bfs, ordering_dfs, mst_axis_ratio
        else:
            return ordering_bfs, ordering_dfs


    ######################################## PROXIMIY/DISTANCE matrix conversion ##########################################
    def _return_proximity_matrix_populated_by_MSTs_avg_prox(self, MST_list, weight_list):
        """Function populates the MSTs from the input list into a proximty matrix, where each MST is weighted according 
        to the appropriate weight from the weight list. The population is done by inserting the edges in an MST into 
        the relevant cells in the proximity matrix, weighted by the appropriate weight.
        (*) Cells in the matrix that do not correspond to any edge will be filled with zero (no proximity).
        (*) Cells in the diagonal of the matrix will be filled with numpy.inf (infinite proximity).

        Parameters
        -------
        :param MST_list: list of networkx.classes.graph.Graph(), a list of the MSTs to be populated into the proximity matrix.
        :param weight_list: list of floats, a list of the weights of each of the MST when populated into the proximity matrix.
            In practice, the weights are defined as the axis ratios of each of the MSTs.

        Returns
        -------
        :param proximity_matrix: numpy.ndarray(), the proximity matrix, populated by the different MSTs.
        """
        assert type(MST_list) == list, "MST_list should be a list"
        assert (numpy.array(weight_list) >= 0).all(), "weights in weight_list should be non-negative"
        assert numpy.fromiter([type(mst) == nx.classes.graph.Graph for mst in MST_list], dtype=bool).all(), "MST_list should contain networkx.classes.graph.Graph objects"

        # take only the N best estimators, if required:
        if self.to_average_N_best_estimators:
            indices = numpy.argsort(weight_list)[::-1][:self.number_of_best_estimators]
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

    def _return_proximity_matrix_populated_by_MSTs_avg_dist(self, MST_list, weight_list):
        """Function populates the MSTs from the input list into a proximty matrix, where each MST is weighted according 
        to the appropriate weight from the weight list. The population is done by inserting the edges in an MST into 
        the relevant cells in the distance matrix (the inverse of the proximity matrix), weighted by the appropriate weight.
        (*) Cells in the matrix that do not correspond to any edge will be filled with zero (no proximity).
        (*) Cells in the diagonal of the matrix will be filled with numpy.inf (infinite proximity).

        Parameters
        -------
        :param MST_list: list of networkx.classes.graph.Graph(), a list of the MSTs to be populated into the proximity matrix.
        :param weight_list: list of floats, a list of the weights of each of the MST when populated into the proximity matrix.
            In practice, the weights are defined as the axis ratios of each of the MSTs.

        Returns
        -------
        :param proximity_matrix: numpy.ndarray(), the proximity matrix, populated by the different MSTs.
        """
        assert type(MST_list) == list, "MST_list should be a list"
        assert (numpy.array(weight_list) >= 0).all(), "weights in weight_list should be non-negative"
        assert numpy.fromiter([type(mst) == nx.classes.graph.Graph for mst in MST_list], dtype=bool).all(), "MST_list should contain networkx.classes.graph.Graph objects"

        # take only the N best estimators, if required:
        if self._to_average_N_best_estimators:
            indices = numpy.argsort(weight_list)[::-1][:self.number_of_best_estimators]
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
        proximity_matrix = self._convert_distance_to_proximity_matrix(distance_matrix)
        return proximity_matrix


    def _convert_proximity_to_distance_matrix(self, proximity_matrix):
        """Function converts the given proximity matrix into a distance matrix and returns it.

        Parameters
        -------
        :param proximity_matrix: numpy.ndarray(), a matrix describing the proximity between the different objects

        Returns
        -------
        :param distance_matrix: numpy.ndarray(), a matrix describing the distance between the different objects. 
            It contains values which are the inverse of the proximity values.
        """
        assert type(proximity_matrix) == numpy.ndarray, "proximity matrix must be numpy.ndarray"
        assert len(proximity_matrix.shape) == 2, "proximity matrix must have 2 dimensions"
        assert proximity_matrix.shape[0] == proximity_matrix.shape[1], "proximity matrix must be NxN matrix"
        assert (~numpy.isnan(proximity_matrix)).all(), "proximity matrix contains nan values"
        assert (~numpy.isneginf(proximity_matrix)).all(), "proximity matrix contains negative infinite values"
        assert (proximity_matrix >= 0).all(), "proximity matrix contains negative values"

        proximity_matrix_copy = numpy.copy(proximity_matrix)
        numpy.fill_diagonal(proximity_matrix_copy, numpy.inf)
        distance_matrix = 1.0 / proximity_matrix_copy
        return distance_matrix

    def _convert_distance_to_proximity_matrix(self, distance_matrix):
        """Function converts the given distance matrix into a proximity matrix and returns it.

        Parameters
        -------
        :param distance_matrix: numpy.ndarray(), a matrix describing the distances between the different objects.

        Returns
        -------
        :param proximity_matrix: numpy.ndarray(), a matrix describing the proximity between the different objects. 
            It contains values which are the inverse of the distance values.
        """
        assert type(distance_matrix) == numpy.ndarray, "distance matrix must be numpy.ndarray"
        assert len(distance_matrix.shape) == 2, "distance matrix must have 2 dimensions"
        assert distance_matrix.shape[0] == distance_matrix.shape[1], "distance matrix must be NxN matrix"
        assert (~numpy.isnan(distance_matrix)).all(), "distance matrix contains nan values"
        assert (~numpy.isneginf(distance_matrix)).all(), "distance matrix contains negative infinite values"
        assert (distance_matrix.round(5) >= 0).all(), "distance matrix contains negative values"

        distance_matrix_copy = numpy.copy(distance_matrix)
        numpy.fill_diagonal(distance_matrix_copy, 0)
        proximity_matrix = 1.0 / distance_matrix_copy
        return proximity_matrix

    ################################################ ALGORITHM FUNCTIONS ##################################################
    def _return_distance_matrix_dictionary_for_estimators_and_scales(self):
        """Function calculates the distance matrices for each distance metric and scale. 
        It uses the list of distance metrics and scales provided by the user. For each metric and scale, function divides 
        the objects into chunks according to the scale, and estimates the distance between the chunks of objects.
    
        Returns
        -------
        :param distance_matrix_dictionary: dict(), a dictionary consisting of all the different distance matrices. The keys 
            of the dictionary are tuples of (estimator_name, scale_value), where estimator_name is a given distance metric
            and scale_value is a given scale. The values of the dictionary are lists consisting of the distance matrices 
            for eahc of the chunks.
        """
        assert type(self.grid) == numpy.ndarray, "grid must be numpy.ndarray"
        assert type(self.objects_list) == numpy.ndarray, "objects_list_normalised must be numpy.ndarray"
        assert ((len(self.grid.shape) == 1) or (len(self.grid.shape) == 2)), "objects can be 1 or 2 dimensional"
        assert (~numpy.isnan(self.grid)).all(), "grid cannot contain nan values"
        assert (~numpy.isinf(self.grid)).all(), "grid cannot contain infinite values"
        assert (~numpy.isneginf(self.grid)).all(), "grid cannot contain negative infinite values"
        assert (~numpy.isnan(self.objects_list)).all(), "objects_list cannot contain nan values"
        assert (~numpy.isinf(self.objects_list)).all(), "objects_list cannot contain infinite values"
        assert (~numpy.isneginf(self.objects_list)).all(), "objects_list cannot contain negative infinite values"
        if len(self.grid.shape) == 1:
            assert (self.grid.shape[0] == self.objects_list.shape[1]), "the grid and the objects must have the same dimensions"
        if len(self.grid.shape) == 2:
            assert ((self.grid.shape[0] == self.objects_list.shape[1]) and (self.grid.shape[1] == self.objects_list.shape[2])), "the grid and the objects must have the same dimensions"
        assert numpy.fromiter([(isinstance(scale_value, int) or type(scale_value) == numpy.int64) for scale_value in numpy.array(self.scale_list).flatten()], dtype=bool).all(), "scale values must all be integers"
        assert numpy.fromiter([estimator_value in ['EMD', 'energy', 'KL', 'L2'] for estimator_value in self.estimator_list], dtype=bool).all(), "estimators must be EMD, energy, KL or L2"

        distance_matrix_dictionary = {}
        for estimator_index, estimator_name in enumerate(self.estimator_list):

            scale_list_for_estimator = self.scale_list[estimator_index]
            for scale_index, scale_value in enumerate(scale_list_for_estimator):

                # printing information and saving it into a log file
                if self.to_print_progress:
                    print("calculating the distance matrices for estimator: %s, scale: %s" % (estimator_name, scale_value))
                if self.file_log != None:
                    self.file_log.write("calculating the distance matrices for estimator: %s, scale: %s\n" % (estimator_name, scale_value))
                    self.file_log.flush()
                
                start_time = time.time()
                # divide the objects into chunks according to the scale
                N_chunks = scale_value
                grid_splitted, objects_list_splitted = self._divide_to_chunks(self.grid, numpy.copy(self.objects_list), N_chunks)
                # construct the distance matrix list for this given scale
                distance_matrix_list = []
                for i in range(len(grid_splitted)):
                    grid_of_chunk = grid_splitted[i]
                    objects_list_of_chunk = objects_list_splitted[i]
                    distance_matrix_of_chunk = self._return_distance_matrix(grid_of_chunk, objects_list_of_chunk, estimator_name)
                    distance_matrix_list.append(distance_matrix_of_chunk)

                if self.to_print_progress: 
                    print("finished calculating this distance matrix list, it took: %s seconds" % str(time.time() - start_time))
                if self.file_log != None:
                    self.file_log.write("finished calculating this distance matrix list, it took: %s seconds \n" % str(time.time() - start_time))
                    self.file_log.flush()

                # add the list of matrices to the dictionary
                distance_matrix_dictionary[(estimator_name, scale_value)] = distance_matrix_list
        return distance_matrix_dictionary

    def _return_weighted_distance_matrix_for_single_estimator_and_scale(self, distance_matrix_list, to_return_axis_ratio_list=True, to_return_sequence_list=True):
        """Function calculates the weighted distance matrix for a single metric and scale. 
        Function takes as an input a list of distance matrices, which correspond to the different chunks at a given scale.
        Function orders the spectra according to each chunk and measures the axis ratio which serves as a weight of each sequence.
        Function then performs a weighted average to return a single distance matrix, according to the axis ratio.

        Parameters
        -------
        :param distance_matrix_list: list of numpy.ndarray(), a list of distance matrices for each chunk.
        :param to_return_axis_ratio_list: boolean (default=True), whether to return a list of axis ratios per chunk.
        :param to_return_sequence_list: boolean (default=True), whether to return a list of the sequences per chunk.

        Returns
        -------
        :param weighted_distance_matrix: numpy.ndarray(), the weighted distance matrix over the different chunks
        :param ordering_list (optional): list of lists, a list that contains the sequences for each different chunk
        :param axis_ratio_list (optional): list of floats, a list that contains the axis ratios for each different chunk
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

        axis_ratio_list = []
        ordering_list = []
        for chunk_index in range(len(distance_matrix_list)):
            distance_matrix_of_chunk = distance_matrix_list[chunk_index]
            ordering_bfs, ordering_dfs, mst_axis_ratio, mst = self._apply_MST_and_return_BFS_DFS_ordering(distance_matrix_of_chunk, return_axis_ratio=True, return_MST=True)
            axis_ratio_list.append(mst_axis_ratio)
            ordering_list.append(ordering_bfs)
        axis_ratio_list = numpy.array(axis_ratio_list)

        # now take the weighted average to the list according to the weights you calculated
        weighted_distance_matrix = numpy.average(distance_matrix_list, axis=0, weights=axis_ratio_list)
        if to_return_axis_ratio_list and to_return_sequence_list:
            return weighted_distance_matrix, ordering_list, axis_ratio_list
        elif to_return_axis_ratio_list and not to_return_sequence_list:
            return weighted_distance_matrix, ordering_list
        else:
            return weighted_distance_matrix


