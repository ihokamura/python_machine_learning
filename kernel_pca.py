"""
kernel PCA transformer
"""

import numpy as np
from scipy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform


class RBFKernelPCA:
    """
    RBF kernel PCA (principal component analysis) transformer

    # Parameters
    -----
    * n_components : int
        number of components to be extracted
    * gamma : float
        hyperparameter of radial basis function

    # Attributes
    -----
    * lambdas_ : array-like, shape = (n_samples, )
        eigenvalues of centraized kernel matrix

    * alphas_ : array-like, shape = (n_samples, n_components)
        eigenvectors of centraized kernel matrix

    # Notes
    -----
    * n_samples represents the number of samples.
    * n_features represents the number of features.
    """

    def __init__(self, n_components=1, gamma=1.0):
        self.n_components = n_components
        self.gamma = gamma

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
        # use shortcut expression instead of returning transformer.transform(X)
        return transformer.alphas_ * np.sqrt(transformer.lambdas_)

    def fit(self, X):
        """
        fit to the training data

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data

        # Returns
        -----
        * self : RBFKernelPCA
            RBFKernelPCA transformer after fit
        """

        # copy training data (referred in `transform()`)
        self.X_fit_ = X.copy()
        self.n_samples = X.shape[0]
        self.one_n = np.ones((self.n_samples, self.n_samples)) / self.n_samples

        # compute square of Euclidean distance for each pair of data points
        sq_dist = squareform(pdist(X, 'sqeuclidean'))

        # compute centralized kernel matrix
        K = exp(-self.gamma * sq_dist)
        K = K - self.one_n@K - K@self.one_n + self.one_n@K@self.one_n

        # compute eigenvectors and sort them in descending order of corresponding eigenvalues
        eigen_values, eigen_vectors = eigh(K)
        eigen_values, eigen_vectors = eigen_values[::-1], eigen_vectors[:, ::-1]

        # save eigenvalues and eigenvectors
        self.lambdas_ = eigen_values[:self.n_components]
        self.alphas_ = np.column_stack(list(eigen_vectors[:, i] for i in range(self.n_components)))

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

        X_transformed = []
        for x in X:
            sq_dist = np.array([np.sum((x - x_sample)**2) for x_sample in self.X_fit_])
            K = np.exp(-self.gamma * sq_dist)
            K = K - self.one_n@K
            x_transformed = K @ (self.alphas_/self.lambdas_)
            X_transformed.append(x_transformed)

        return np.array(X_transformed)
