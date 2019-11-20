"""
show sample of how to use KMeans
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from k_means import KMeans


def main():
    # prepare sample data
    centers = 3
    X, _ = make_blobs(
        n_samples=150,
        n_features=2,
        centers=centers,
        cluster_std=0.5,
        shuffle=True,
        random_state=0)

    # fit clusterings
    clusterings = [
        KMeans(
            n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1.0e-4,
            random_state=1),
        KMeans(
            n_clusters=3,
            init='k-means++',
            n_init=1,
            max_iter=300,
            tol=1.0e-4,
            random_state=1)
    ]
    names = ['k-means', 'k-means++']

    for clustering, name in zip(clusterings, names):
        # predict centroid of clusters and label of each data points
        y_pred = clustering.fit_predict(X)
        # plot predicted labels
        for i in range(centers):
            Xi = X[y_pred == i]
            plt.scatter(
                Xi[:, 0], Xi[:, 1],
                marker='o', edgecolor='black', label='cluster {0}'.format(i + 1))
        # plot centroids
        plt.scatter(
            clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1],
            marker='*', edgecolor='black', label='centroids')
        # set plot area
        plt.grid()
        plt.legend()
        plt.title(name)
        plt.tight_layout()
        plt.show()

        # show attributes
        print('inertia:{0}'.format(clustering.inertia_))
        print('iteration times:{0}'.format(clustering.n_iter_))


if __name__ == '__main__':
    main()
