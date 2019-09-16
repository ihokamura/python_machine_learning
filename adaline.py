"""
ADALINE classifier
"""

import numpy as np


class Adaline:
    """
    ADALINE (adaptive linear neuron) classifier

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
        * self : Adaline
            ADALINE classifier after fit
        """

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

        # define this function to consider a general activation function in later
        return x

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

        # use np.where() considering a general case
        return np.where(self.activate(self.net_input(x)) >= 0.0, 1, -1)


class AdalineGD(Adaline):
    """
    ADALINE (adaptive linear neuron) classifier implemented with gradiend descend method

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
        * self : AdalineGD
            ADALINE classifier after fit
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.costs_ = []

        for _ in range(self.n_iter):
            # update weights
            error = y - self.activate(self.net_input(X))
            self.w_[1:] += self.eta * (error @ X)
            self.w_[0] += self.eta * error.sum()
            # log cost
            self.costs_.append(0.5 * np.sum(error**2))

        return self


class AdalineSGD(Adaline):
    """
    ADALINE (adaptive linear neuron) classifier implemented with stochastic gradiend descend method

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
    * w_ : 1-d array
        weight after fit
    * costs_ : list
        costs (mean square error) at each epoch

    # Notes
    -----
    * n_samples represents the number of samples.
    * n_features represents the number of features.
    """

    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=1):
        super(AdalineSGD, self).__init__(eta=eta, n_iter=n_iter, random_state=random_state)
        self.shuffle = shuffle
        self.w_initialized = False

    def fit(self, X, y):
        """
        fit to the training data

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data
        * y : array-like, shape = (n_samples, )
            target variable

        Returns
        -----
        * self : AdalineSGD
            ADALINE classifier after fit
        """

        self._initialize_weights(X.shape[1])
        self.costs_ = []

        for _ in range(self.n_iter):
            # shuffle training data if required
            if self.shuffle:
                X, y = self._shuffle(X, y)
            # update weights
            sum_cost = 0.0
            for xi, target in zip(X, y):
                cost = self._update_weights(xi, target)
                sum_cost += cost
            # log cost
            self.costs_.append(sum_cost / len(y))

        return self

    def partial_fit(self, X, y):
        """
        fit training data without re-initializing weights

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data
        * y : array-like, shape = (n_samples, )
            target variable

        # Returns
        -----
        * self : AdalineSGD
            ADALINE classifier after fit
        """

        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        for xi, target in zip(X, y):
            self._update_weights(xi, target)

        return self
            
    def _shuffle(self, X, y):
        """
        shuffle training data

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data before shuffle
        * y : array-like, shape = (n_samples, )
            target variable before shuffle

        # Returns
        -----
        * X : array-like, shape = (n_samples, n_features)
            training data after shuffle
        * y : array-like, shape = (n_samples, )
            target variable after shuffle
        """

        r = self.rgen_.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, n_features):
        """
        initialize weights

        # Parameters
        -----
        * n_features : int
            number of features

        # Returns
        -----
        * None
        """

        self.rgen_ = np.random.RandomState(self.random_state)
        self.w_ = self.rgen_.normal(loc=0.0, scale=0.01, size=1 + n_features)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """
        update weights

        # Parameters
        -----
        * xi : array-like, shape = (n_features, )
            sample of training data
        * target : float
            sample of target variable

        # Returns
        -----
        * cost : float
            cost from the given sample
        """

        error = target - self.activate(self.net_input(xi))
        self.w_[1:] += self.eta * error * xi
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2

        return cost
