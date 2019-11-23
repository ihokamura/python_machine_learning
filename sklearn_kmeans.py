"""
show sample of how to use KMeans
"""

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples


def main():
    show_clustering()
    show_elbow()
    show_silhouette()


def show_clustering():
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
            n_init=10,
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


def show_elbow():
    # prepare sample data
    centers = 3
    X, _ = make_blobs(
        n_samples=150,
        n_features=2,
        centers=centers,
        cluster_std=0.5,
        shuffle=True,
        random_state=0)

    # compute inertia for different number of clusters
    n_clusters_range = range(1, 11)
    inertia = []
    for n_clusters in n_clusters_range:
        # fit clustering
        clustering = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1.0e-4,
            random_state=1)
        clustering.fit(X)
        inertia.append(clustering.inertia_)

    # plot inertia
    plt.plot(n_clusters_range, inertia, marker='o')
    plt.xlabel('number of clusters')
    plt.ylabel('inertia')
    plt.tight_layout()
    plt.show()


def show_silhouette():
    # prepare sample data
    centers = 3
    X, _ = make_blobs(
        n_samples=150,
        n_features=2,
        centers=centers,
        cluster_std=0.5,
        shuffle=True,
        random_state=0)

    for n_clusters in range(2, 5):
        # fit clustering and predict labels
        clustering = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1.0e-4,
            random_state=1)
        y = clustering.fit_predict(X)

        # compute silhouette coefficients
        cluster_labels = np.unique(y)
        n_clusters = len(cluster_labels)
        silhouette_coefficients = silhouette_samples(X, y, metric='euclidean')

        # show silhouette plot
        yticks = []
        y_lower, y_upper = 0, 0
        for i, c in enumerate(cluster_labels):
            values = np.sort(silhouette_coefficients[y == c])
            color = cm.jet(i / n_clusters)
            y_upper += len(values)
            plt.barh(
                range(y_lower, y_upper), values,
                height=1.0, color=color, edgecolor='none')
            yticks.append((y_lower + y_upper) / 2)
            y_lower += len(values)
        plt.axvline(np.mean(silhouette_coefficients), color='black', linestyle='--')
        plt.yticks(yticks, cluster_labels + 1)
        plt.title('k = {0}'.format(n_clusters))
        plt.xlabel('silhouette coefficient')
        plt.ylabel('cluster')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
