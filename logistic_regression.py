"""
logistic regression classifier
"""

import abc

import numpy as np


class LogisticRegression(abc.ABC):
    """
    logistic regression classifier

    # Parameters
    -----
    * eta : float
        learning rate (greather than 0.0 and less than or equal to 1.0)
    * n_iter : int
        training times of training data
    * random_state : int
        random generator seed used to initialize weights

    # Attributes
    -----
    w_ : 1-d array
        weight after fit
    costs_ : list
        costs (mean square error) at each epoch

    # Notes
    -----
    * n_samples represents the number of samples.
    * n_features represents the number of features.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    @abc.abstractmethod
    def fit(self, X, y):
        """
        fit to the training data

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data
        * y : array-like, shape = (n_samples, )
            target variable

        # Returns
        -----
        * self : LogisticRegression
            logistic regression classifier after fit
        """

        return self

    @abc.abstractmethod
    def net_input(self, X):
        """
        compute net input

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data

        # Returns
        -----
        * _ : array-like, shape = (n_samples, )
            net input, which means linear combination of weights and sample data (including bias)
        """

    def activate(self, x):
        """
        compute activation function

        # Parameters
        -----
        * x : array-like, shape = (n_samples, )
            input of activation function for each sample

        # Returns
        -----
        * _ : array-like, shape = (n_samples, )
            value of activation function for each sample
        """

        bound = 250
        return 1.0 / (1.0 + np.exp(-np.clip(x, -bound, bound)))

    def predict(self, x):
        """
        return class label after one step

        # Parameters
        -----
        * x : array-like, shape = (n_features, )
            data of which label is predicted

        # Returns
        -----
        * out : int
            class label after one step
        """

        # return 1 if \phi(z) >= 0.5 / return 0 otherwise
        return np.where(self.net_input(x) >= 0.0, 1, 0)


class LogisticRegressionGD(LogisticRegression):
    """
    logistic regression classifier

    # Parameters
    -----
    * eta : float
        learning rate (greather than 0.0 and less than or equal to 1.0)
    * n_iter : int
        training times of training data
    * random_state : int
        random generator seed used to initialize weights

    # Attributes
    -----
    w_ : 1-d array
        weight after fit
    costs_ : list
        costs (mean square error) at each epoch

    # Notes
    -----
    * n_samples represents the number of samples.
    * n_features represents the number of features.
    """

    def fit(self, X, y):
        """
        fit to the training data

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data
        * y : array-like, shape = (n_samples, )
            target variable

        # Returns
        -----
        * self : LogisticRegressionGD
            logistic regression classifier after fit
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.costs_ = []

        for _ in range(self.n_iter):
            # update weights
            output = self.activate(self.net_input(X))
            error = y - output
            self.w_[1:] += self.eta * (X.T @ error)
            self.w_[0] += self.eta * np.sum(error)
            # log cost
            cost = -(y @ np.log(output) + (1 - y) @ np.log(1 - output))
            self.costs_.append(cost)

        return self

    def net_input(self, X):
        """
        compute net input

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data

        # Returns
        -----
        * _ : array-like, shape = (n_samples, )
            net input, which means linear combination of weights and sample data (including bias)
        """

        return X @ self.w_[1:] + self.w_[0]
