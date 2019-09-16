"""
perceptron classifier
"""

import numpy as np


class Perceptron:
    """
    perceptron classifier

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
    * w_ : array-like, shape = (n_features, )
        weights after fit
    * errors_ : list
        number 
        of errors after each epoch

    # Notes
    -----
    * n_samples represents the number of samples.
    * n_features represents the number of features.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

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
        * self : Perceptron
            perceptron classifier after fit
        """

        # initialize weights and errors
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        # train for given times
        for _ in range(self.n_iter):
            errors = 0
            for xi, yi in zip(X, y):
                # update weights
                delta = self.eta * (yi - self.predict(xi))
                self.w_[1:] += delta * xi
                self.w_[0] += delta
                # accumulate error
                errors += int(delta != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self, x):
        """
        compute net input

        # Parameters
        -----
        * x: array-like, shape = (n_features, )
            sample of training data

        # Returns
        -----
        * _ : float
             net input
        """

        return (x @ self.w_[1:] + self.w_[0])

    def predict(self, x):
        """
        return class label after one step

        # Parameters
        -----
        * x : array-like, shape = (n_features, )
             data of which label is predicted

        # Returns
        -----
        * _ : int
             class label after one step
        """

        # use np.where() to consider a general case in later
        return np.where(self.net_input(x) >= 0.0, 1, -1)
