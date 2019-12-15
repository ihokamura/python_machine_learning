"""
utility for CNN
"""

import numpy as np


def generate_batch(X, y, batch_size=64, shuffle=False, random_seed=None):
    """
    generate minibatch

    # Parameters
    -----
    * X : array-like, shape = (n_samples, n_features)
        training data
    * y : array-like, shape = (n_samples, )
        target variable
    * batch_size : int
        size of minibatch
    * shuffle : bool
        indicator to check if it is necessary to shuffle data
    * random_seed : int or `None`
        random generator seed

    # Yields
    -----
    * _ : array-like, shape = (batch_size, n_features)
        minibatch of training data
    * _ : array-like, shape = (batch_size, )
        minibatch of target variable

    # Notes
    -----
    * n_samples represents the number of samples.
    * n_features represents the number of features.
    """

    index = np.arange(y.shape[0])
    if shuffle:
        np.random.RandomState(random_seed).shuffle(index)
        X = X[index]
        y = y[index]

    for i in range(0, X.shape[0], batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]
