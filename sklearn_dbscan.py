"""
show sample of how to use DBSCAN
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.datasets import make_moons


def main():
    # prepare dataset
    X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

    # compare clustering algorithms
    clusterings = [
        KMeans(n_clusters=2, random_state=0),
        AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete'),
        DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
    ]
    names = ['k-means', 'agglomerative clustering', 'DBSCAN']
    for clustering, name in zip(clusterings, names):
        # fit to dataset and predict cluster labels
        y = clustering.fit_predict(X)

        # plot clusters
        plt.scatter(X[y == 0, 0], X[y == 0, 1], edgecolor='black', marker='o', label='cluster 1')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], edgecolor='black', marker='s', label='cluster 2')
        plt.legend()
        plt.title(name)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
