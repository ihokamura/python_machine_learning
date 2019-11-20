"""
k-means clustering
"""

import numpy as np


class KMeans:
    """
    k-means clustering

    # Parameters
    -----
    * n_clusters : int
        number of clusters

    * init : str
        strategy to choose initial centroids
        One of the following must be specified:
            * 'random' : choose initial centroids at random
            * 'k-means++' : choose initial centroids according to k-means++ method

    * n_init : int
        number of times to run k-means algorithm with different centroid seeds

    * max_iter : int
        maximum number of iterations

    * tol : float
        tolerance of inertia for convergence check

    * random_state : int
        random generator seed used to initialize centroids

    # Attributes
    -----
    * cluster_centers_ : array-like, shape = (n_clusters, n_features)
        centroids of clusters

    * inertia_ : float
        sum of squared distances of data points to the nearest centroid

    * n_iter : int
        number of iterations

    # Notes
    -----
    * n_samples represents the number of samples.
    * n_features represents the number of features.
    """

    def __init__(self, n_clusters=2, init='random', n_init=10, max_iter=100, tol=1.0e-4, random_state=1):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_predict(self, X):
        """
        fit to and predict the training data

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data

        # Returns
        -----
        * _ : array-like, shape = (n_samples, )
            predicted class labels
        """

        clustering = self.fit(X)
        return clustering.predict(X)

    def fit(self, X):
        """
        fit to the training data

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data

        # Returns
        -----
        * self : KMeans
            k-means clustering after fit
        """

        # run k-means algorithm with different centroid seeds
        self._rgen = np.random.RandomState(self.random_state)
        results = list(self._fit_once(X) for _ in range(self.n_init))

        # choose the best centroids and save attributes
        best_index = np.argmin(list(result[0] for result in results))
        self.inertia_, self.n_iter_, self.cluster_centers_ = results[best_index]

        return self

    def predict(self, X):
        """
        predict the training data

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data

        # Returns
        -----
        * _ : array-like, shape = (n_samples, )
            predicted class labels
        """

        def _generate_labels():
            for x in X:
                metrics = list(np.sum((x - mu)**2) for mu in self.cluster_centers_)
                yield np.argmin(metrics)

        return np.array(list(_generate_labels()))

    def _fit_once(self, X):
        """
        fit to the training data from a set of initial centroids

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data

        # Returns
        -----
        * inertia : float
            sum of squared distances of data points to the nearest centroid

        * n_iter : int
            number of iterations

        * centroids : array-like, shape = (n_clusters, n_features)
            centroids of clusters
        """

        # choose initial centroids
        centroids = self._init_centroids(X)

        n_samples = X.shape[0]
        cluster_labels = -np.ones((n_samples,))
        for n_iter in range(self.n_clusters):
            change = False
            # map each data points to the nearest centroid
            for i in range(n_samples):
                metrics = list(np.sum((X[i] - mu)**2) for mu in centroids)
                label = np.argmin(metrics)
                if cluster_labels[i] != label:
                    change = True
                    cluster_labels[i] = label

            if change is True:
                # check convergence
                inertia = sum(np.sum((X[cluster_labels == j] - centroids[j])**2) for j in range(self.n_clusters))
                if inertia < self.tol:
                    break

                # update centroids
                for j in range(self.n_clusters):
                    points = X[cluster_labels == j]
                    if len(points) == 0:
                        # search the farthest data point to be a new centroid
                        metrics = list(np.sum((X[i] - centroids[j])**2) for i in range(n_samples))
                        centroids[j] = X[np.argmin(metrics)]
                    else:
                        centroids[j] = np.mean(points, axis=0)
            else:
                # end the iteration since no labels have changed
                inertia = sum(np.sum((X[cluster_labels == j] - centroids[j])**2) for j in range(self.n_clusters))
                break

        # return inertia , iteration times and centroids
        return inertia, n_iter, centroids

    def _init_centroids(self, X):
        """
        choose initial centroids

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data

        # Returns
        -----
        * centroids : array-like, shape = (n_clusters, n_features)
            initial centroids of clusters
        """

        n_samples = X.shape[0]
        if self.init == 'random':
            # choose initial centroids at random
            return X[self._rgen.choice(n_samples, size=self.n_clusters)]
        else:
            # choose initial centroids according to k-means++ method
            centroid_indices = [self._rgen.choice(n_samples)]
            for _ in range(self.n_clusters - 1):
                distances = np.array(list(np.min(list(np.sum((X[i] - X[j])**2) for j in centroid_indices)) for i in range(n_samples)))
                centroid_indices.append(self._rgen.choice(n_samples, p=distances / np.sum(distances)))

            return X[centroid_indices]
