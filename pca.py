"""
PCA transformer
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class PCA:
    """
    PCA (principal component analysis) transformer

    # Parameters
    -----
    * n_components : int
        number of components to be extracted

    # Attributes
    -----
    * components_ : array-like, shape = (n_components, n_features)
        principal components
    
    * explained_variance_ : array-like, shape (n_components,)
        variance explained by each components

    # Notes
    -----
    * n_samples represents the number of samples.
    * n_features represents the number of features.
    """

    def __init__(self, n_components=1):
        self.n_components = n_components
        self._sc = StandardScaler()

    def fit_transform(self, X):
        """
        fit to and transform the training data

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data

        # Returns
        -----
        * _ : array-like, shape = (n_samples, n_components)
            transformed data
        """

        transformer = self.fit(X)
        return transformer.transform(X)

    def fit(self, X):
        """
        fit to the training data

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data

        # Returns
        -----
        * self : PCA
            PCA transformer after fit
        """

        self._sc.fit_transform(X)
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

        covariance = np.cov(X.T)
        eigen_values, eigen_vectors = np.linalg.eig(covariance)
        eigen_pairs = list((eigen_values[i], eigen_vectors[:, i]) for i in range(len(eigen_values)))
        eigen_pairs.sort(key=(lambda pair: np.abs(pair[0])), reverse=True)

        self.components_ = np.vstack(list(eigen_pairs[i][1] for i in range(self.n_components)))
        self.explained_variance_ = np.array(list(eigen_values[i] for i in range(self.n_components)))

        return X @ self.components_.T
