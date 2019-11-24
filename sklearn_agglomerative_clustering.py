"""
show sample of how to use AgglomerativeClustering
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def main():
    # prepare dataset
    variables = list('XYZ')
    labels = list('ID_{index}'.format(index=i) for i in range(5))
    np.random.seed(123)
    X = 10 * np.random.random_sample((len(labels), len(variables)))

    # execute clustering for different number of clusters
    for n_clusters in range(1, len(labels) + 1):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='complete')
        y = clustering.fit_predict(X)
        print('n_clusters={0} -> {1}'.format(n_clusters, y))


if __name__ == '__main__':
    main()
