# The Sequencer
An algorithm that detects one-dimensional trends (=sequences) in complex datasets. 

## Overview:
The Sequencer is an algorithm that attempts to reveal the main sequence in a dataset, if it exists.  To do so, it reorders objects within a set to produce the most elongated manifold describing their similarities which are measured in a multi-scale manner and using a collection of metrics. To be generic, it combines information from four different distance metrics: the Euclidean Distance, the Kullback-Leibler Divergence, the Monge-Wasserstein or Earth Mover Distance, and the Energy Distance. In addition, it considers different scales by dividing the input data into chunks, and estimating the similarities between objects in each of these chunks. 

The Sequencer uses the shape of the graphs describing the multi-scale similarities. In particular, it uses the fact that continuous trends (sequences) in a dataset lead to more elongated graphs. By defining the elongation of the graphs as a figure of merit, the Sequencer can optimize over its hyper-parameters (the distance metrics and scales) and select the set of metric+scale that are most sensitive to the presence of a sequence in the data. The elongation of the graph is measured using the graph's axis ratio. Thus, the final output of the Sequencer is the detected sequence and its associated axis ratio. Small axis ratio (~1) suggests no clear sequence in the data, and large axis ratio (~N, where N is the number of objects in the sample) suggests that the Sequencer detected a significant sequence in the data. 

The Sequencer is essentially an Unsupervised Dimensionality Reduction algorithm, since a sequence is a one-dimensional embedding of the input dataset. There are several differences between the Sequencer and other dimensionality reduction techniques like tSNE and UMAP. First, the Sequencer can only embed the input dataset into a single dimension, while algorithms like tSNE and UMAP can embed the data into higher dimensions as well (2D or 3D). The main advantage of the Sequencer is that it uses a figure of merit to optimize over its hyper-parameters, while other dimensionality reduction algorithms depend on a set of hyper-parameters and lack a clear figure of merit with which these can be optimized. As a result, the output of other dimensionality reduction algorithms depends on the chosen hyper-parameters, which are often set manually. In our work, we show that in some cases the Sequencer outperforms tSNE and UMAP in finding one-dimensional trends in the data, especially in scientific datasets.

## Authors
* Dalya Baron (Tel Aviv University; dalyabaron@gmail.com)
* Brice Ménard (Johns Hopkins University)

## Requirements
* python 3.7.1
* numpy 1.18.1
* scipy 1.3.1
* networkx 2.4
* 
