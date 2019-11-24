"""
show sample of how to use linkage and dendrogram of scipy
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

import hierarchy


def main():
    # prepare dataset
    variables = list('XYZ')
    labels = list('ID_{index}'.format(index=i) for i in range(5))
    np.random.seed(123)
    X = 10 * np.random.random_sample((len(labels), len(variables)))
    df = pd.DataFrame(X, columns=variables, index=labels)
    print(df)

    # compute distance matrix
    dist = squareform(pdist(X, metric='euclidean'))
    print(pd.DataFrame(dist, columns=labels, index=labels))

    # compute linkage matrix
    clusters = linkage(pdist(X, metric='euclidean'), method='complete')
    #clusters = hierarchy.linkage(squareform(pdist(X, metric='euclidean')), method='complete')
    print(pd.DataFrame(clusters, columns=['label 1', 'label 2', 'distance', 'items'], index=['cluster {0}'.format(i + 1) for i in range(len(clusters))]))

    # plot dendrogram
    dendrogram(clusters, labels=labels)
    plt.ylabel('Euclidean distance')
    plt.tight_layout()
    plt.show()

    # plot dendrogram with heat map
    # plot dendrogram
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    ax_dendr = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    dendr = dendrogram(clusters, orientation='left')
    ax_dendr.set_xticks([])
    ax_dendr.set_yticks([])
    for i in ax_dendr.spines.values():
        i.set_visible(False)
    # plot heat map representing features of dataset points
    df_clusters = df.iloc[dendr['leaves'][::-1]]
    ax_hm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
    cax = ax_hm.matshow(df_clusters, interpolation='nearest', cmap='hot_r')
    ax_hm.set_xticklabels([''] + list(df_clusters.columns))
    ax_hm.set_yticklabels([''] + list(df_clusters.index))
    fig.colorbar(cax)
    plt.show()


if __name__ == '__main__':
    main()
