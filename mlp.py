"""
multilayer perceptron classifier
"""

import numpy as np


class MLP:
    """
    MLP (multilayer perceptron) classifier

    # Parameters
    -----
    * n_hidden : int
        number of units in the hidden layer
    * l2 : float
        L2-regularization parameter
    * epochs : int
        number of epochs
    * eta : float
        learning rate (greather than 0.0 and less than or equal to 1.0)
    * shuffle : bool
        indicator to check if it is necessary to shuffle samples before updating weights at each epochs
    * minibatch_size : int
        number of samples for each mini-batch
    * seed : int or None
        random generator seed used to initialize weights and shuffle

    # Attributes
    -----
    * costs_ : list
        costs at each epoch
    * accuracys_ : list
        accuracy at each epoch

    # Notes
    -----
    * n_samples represents the number of samples.
    * n_features represents the number of features.
    """

    def __init__(self, n_hidden=30, l2=0.0, epochs=100, eta=0.001, shuffle=True, minibatch_size=1, seed=None):
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        self.random = np.random.RandomState(seed)

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
        * self : MLP
            MLP classifier after fit
        """

        n_samples, n_features = X.shape
        n_hidden = self.n_hidden
        n_labels = len(np.unique(y))
        y_encoded = self._encode_onehot(y, n_labels)

        # initialize weights
        self.b_h = np.zeros(n_hidden)
        self.W_h = np.random.normal(loc=0.0, scale=0.1, size=(n_features, n_hidden))
        self.b_o = np.zeros(n_labels)
        self.W_o = np.random.normal(loc=0.0, scale=0.1, size=(n_hidden, n_labels))

        accuracys = []
        costs = []
        # iterate over epochs
        for _ in range(self.epochs):
            indices = np.arange(n_samples)
            if self.shuffle:
                self.random.shuffle(indices)

            # iterate over mini-batches
            for start_index in range(0, n_samples - self.minibatch_size + 1, self.minibatch_size):
                # set indices of mini-batch
                batch_index = indices[start_index:start_index + self.minibatch_size]

                # forward propagation
                _, A_h, _, A_o = self._feed_forward(X[batch_index])

                # backward propagation
                # from output layer to hidden layer
                S_o = A_o - y_encoded[batch_index]
                dW_o = A_h.T @ S_o
                db_o = np.sum(S_o, axis=0)
                self.W_o -= self.eta*(dW_o + self.l2*self.W_o)
                self.b_o -= self.eta *db_o

                # from hidden layer to input layer
                S_h = S_o @ self.W_o.T * A_h * (1.0 - A_h)
                dW_h = X[batch_index].T @ S_h
                db_h = np.sum(S_h, axis=0)
                self.W_h -= self.eta*(dW_h + self.l2*self.W_h)
                self.b_h -= self.eta * db_h

            # log history
            cost = self._compute_cost(y_encoded[batch_index], A_o)
            costs.append(cost)
            y_pred = self.predict(X)
            accuracy = np.sum(y_pred==y) / n_samples
            accuracys.append(accuracy)

        # save history
        self.costs_ = costs
        self.accuracys_ = accuracys

        return self

    def predict(self, X):
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

        _, _, Z_o, _ = self._feed_forward(X)

        return np.argmax(Z_o, axis=1)

    def _encode_onehot(self, y, n_labels):
        """
        encode class labels in one-hot representation

        # Parameters
        -----
        * y : array-like, shape = (n_samples, )
            target variables
        * n_labels : int
            number of class labels

        # Returns
        -----
        * onehot : array-like, shape = (n_samples, n_labels)
            one-hot representation of class labels
        """

        onehot = np.zeros((y.shape[0], n_labels))
        for index, value in enumerate(y.astype(int)):
            onehot[index, value] = 1.0

        return onehot

    def _sigmoid(self, x):
        """
        compute sigmoid function

        # Parameters
        -----
        * x : array-like
            input of sigmoid function for each sample

        # Returns
        -----
        * _ : array-like
            value of sigmoid function for each sample
        """

        bound = 250
        return 1.0 / (1.0 + np.exp(-np.clip(x, -bound, bound)))

    def _feed_forward(self, X):
        """
        execute feed forward propagation

        # Parameters
        -----
        * X : array-like, shape = (minibatch_size, n_features)
            input of MLP

        # Returns
        -----
        * Z_h : array-like
            input of hidden layer
        * A_h : array-like
            output of hidden layer
        * Z_o : array-like
            input of output layer
        * A_o : array-like
            output of output layer (i.e. prediction)
        """

        # from input layer to hidden layer
        Z_h = X@self.W_h + self.b_h
        A_h = self._sigmoid(Z_h)

        # from hidden layer to output layer
        Z_o = A_h@self.W_o + self.b_o
        A_o = self._sigmoid(Z_o)

        return Z_h, A_h, Z_o, A_o

    def _compute_cost(self, y_encoded, y_pred):
        """
        compute cost

        # Parameters
        -----
        * X : array-like, shape = (minibatch_size, n_features)
            input of MLP

        # Returns
        -----
        * _ : float
            value of cost function
        """

        return -(np.sum(y_encoded*np.log(y_pred) + (1.0 - y_encoded)*np.log(1.0 - y_pred)) + self.l2*(np.sum(self.W_h**2) + np.sum(self.W_o**2)))
