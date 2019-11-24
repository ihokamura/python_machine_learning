"""
hierarchy clustering
"""

from itertools import combinations, product

import numpy as np


def _single_linkage_distance(y, cluster1, cluster2):
    """
    compute distance between clusters by single linkage method

    # Parameters
    -----
    * y : array-like, shape = (n_samples, n_samples)
        distance matrix over all the points, whose (i, j) element is equal to distance between i-th and j-th points
    * cluster1 : array-like
        indices of points in the first cluster
    * cluster2 : array-like
        indices of points in the second cluster

    # Returns
    -----
    * _ : float
        distance between clusters by single linkage method

    # Notes
    -----
    * n_samples represents the number of samples.
    """

    return min(y[i, j] for (i, j) in product(cluster1, cluster2))


def _complete_linkage_distance(y, cluster1, cluster2):
    """
    compute distance between clusters by complete linkage method

    # Parameters
    -----
    * y : array-like, shape = (n_samples, n_samples)
        distance matrix over all the points, whose (i, j) element is equal to distance between i-th and j-th points
    * cluster1 : array-like
        indices of points in the first cluster
    * cluster2 : array-like
        indices of points in the second cluster

    # Returns
    -----
    * _ : float
        distance between clusters by complete linkage method

    # Notes
    -----
    * n_samples represents the number of samples.
    """

    return max(y[i, j] for (i, j) in product(cluster1, cluster2))


# dictionary of functions to compute distance between clusters
_CLUSTER_DISTANCE = {
    'single':_single_linkage_distance,
    'complete':_complete_linkage_distance
}


def linkage(y, method='single'):
    """
    compute linkage matrix

    # Parameters
    -----
    * y : array-like, shape = (n_samples, n_samples)
        distance matrix over all the points, whose (i, j) element is equal to distance between i-th and j-th points
    * method : str
        method to compute distance between clusters
        One of the following must be specified.
        * 'single' : single linkage method
        * 'complete' : complete linkage method

    # Returns
    -----
    * _ : array-like, shape = (n_samples - 1, 4)
        linkage matrix, whose columns consist of
        * first label of aggregated clusters
        * second label of aggregated clusters
        * distance between aggregated clusters
        * number of points in aggregated clusters

    # Notes
    -----
    * n_samples represents the number of samples.
    """

    # assign distance method
    dist_method = _CLUSTER_DISTANCE[method]

    # initialize lists to manage clustering
    n_samples = y.shape[0]
    clusters = list([i] for i in range(n_samples)) # indices of points in clusters
    aggregation_flag = [False] * n_samples # flags to indicate if a cluster has already been aggregated

    # construct linkage matrix
    linkage_matrix = []
    while True:
        # search the nearest pair of clusters
        min_dist = np.inf
        for i, j in combinations(range(len(clusters)), 2):
            if not (aggregation_flag[i] or aggregation_flag[j]):
                cluster1, cluster2 = clusters[i], clusters[j]
                dist = dist_method(y, cluster1, cluster2)
                if dist < min_dist:
                    min_dist = dist
                    aggregated_cluster = (cluster1, cluster2)
                    cluster1_index, cluster2_index = i, j

        # aggregate the clusters
        aggregation_flag[cluster1_index] = True
        aggregation_flag[cluster2_index] = True
        new_cluster = aggregated_cluster[0] + aggregated_cluster[1]
        clusters.append(new_cluster)
        aggregation_flag.append(False)

        # make a new row of linkage matrix
        row = [cluster1_index, cluster2_index, min_dist, len(new_cluster)]
        linkage_matrix.append(row)

        if len(new_cluster) == n_samples:
            # all the clusters have been aggregated
            break

    return np.array(linkage_matrix)
