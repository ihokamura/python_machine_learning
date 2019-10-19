"""
LDA classifier
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class LDA:
    """
    LDA (linear discriminant analysis) classifier

    # Parameters
    -----
    * n_components : int
        number of components to be extracted

    # Attributes
    -----
    * coef_ : array-like, shape = (n_components, n_features)
        weight vectors

    * explained_variance_ratio_ : array-like, shape (n_components,)
        ratio of variance explained by each components

    # Notes
    -----
    * n_samples represents the number of samples.
    * n_features represents the number of features.
    """

    def __init__(self, n_components=1):
        self.n_components = n_components
        self._sc = StandardScaler()

    def fit_transform(self, X, y):
        """
        fit to and transform the training data

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data
        * y : array-like, shape = (n_samples,)
            target variable

        # Returns
        -----
        * _ : array-like, shape = (n_samples, n_components)
            transformed data
        """

        classifier = self.fit(X, y)
        return classifier.transform(X)

    def fit(self, X, y):
        """
        fit to the training data

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data
        * y : array-like, shape = (n_samples,)
            target variable

        # Returns
        -----
        * self : LDA
            LDA transformer after fit
        """

        X = self._sc.fit_transform(X)
        n_features = X.shape[1]
        labels = np.unique(y)

        # compute within-class scatter matrix
        Sw = np.zeros((n_features, n_features))
        for i in labels:
            Sw += np.cov(X[y == i].T)

        # compute between-class scatter matrix
        mean = np.mean(X, axis=0)
        Sb = np.zeros((n_features, n_features))
        for i in labels:
            Xi = X[y == i]
            mi = (np.mean(Xi, axis=0) - mean).reshape(n_features, 1)
            Sb += Xi.shape[0]*(mi @ mi.T)

        eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
        eigen_pairs = list((np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values)))
        eigen_pairs.sort(key=(lambda pair: pair[0]), reverse=True)

        self.coef_ = np.vstack(list(eigen_pairs[i][1].real for i in range(self.n_components)))
        experienced_variance = list(pair[0] for pair in eigen_pairs)
        total_variance = np.sum(experienced_variance)
        self.explained_variance_ratio_ = np.array(list(experienced_variance[i]/total_variance for i in range(self.n_components)))

        return self

    def transform(self, X):
        """
        transform the training data

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data

        # Returns
        -----
        * _ : array-like, shape = (n_samples, n_components)
            transformed data
        """

        return X @ self.coef_.T
