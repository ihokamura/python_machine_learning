"""
SBS algorithm
"""

from itertools import combinations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SBS:
    """
    SBS (sequential backward selection) algorithm

    # Parameters
    -----
    * estimator : object
        estimator, which must implement the following methods
            * fit()
            * transform()
    * k_features : int
        number of features to be selected
    * scoring : function
        method to score features
    * test_size : float
        ratio of test data (between 0 and 1)
    * random_state : int
        random generator seed

    # Attributes
    -----
    * indices_ : tuple
        tuple of indices of features which achieves the best score
    * subsets_ : list
        list of best-scoring indices

    # Notes
    -----
    * n_samples represents the number of samples.
    * n_features represents the number of features.
    """

    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.estimator = estimator
        self.k_features = k_features
        self.scoreing = scoring
        self.test_size = test_size
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
        * self : SBS
            SBS object after fit
        """

        # split training dataset and test dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

        n_features = X_train.shape[1]
        self.indices_ = tuple(range(n_features))
        self.subsets_ = [self.indices_]
        score = self._calculate_score(X_train, X_test, y_train, y_test, self.indices_)
        self.scores_ = [score]

        for dimension in range(n_features, self.k_features, -1):
            # calculate score for all the pairs of indices with a given dimension
            scores = []
            subsets = []
            for indices in combinations(self.indices_, dimension - 1):
                score = self._calculate_score(X_train, X_test, y_train, y_test, indices)
                scores.append(score)
                subsets.append(indices)

            # search and save pair of indices which achieves the best score
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            self.scores_.append(scores[best])

        return self

    def transform(self, X):
        """
        transform features

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            dataset to be transformed

        # Returns
        -----
        * _ : array-like, shape = (n_samples, self.k_features)
            dataset restricted to selected features
        """

        return X[:, self.indices_]

    def _calculate_score(self, X_train, X_test, y_train, y_test, indices):
        """
        calculate score

        # Parameters
        -----
        * X_train : array-like, shape = (n_features, n_samples*(1 - self.test_size))
            training dataset (input)
        * X_test : array-like, shape = (n_features, n_samples*self.test_size)
            test dataset (input)
        * y_train : array-like, shape=(n_samples*(1 - self.test_size), )
            training dataset (output)
        * y_test : array-like, shape=(n_samples*self.test_size, )
            test dataset (output)
        * indices : tuple
            indices of selected features

        # Returns
        -----
        * _ : float
             score of the estimator with selected features
        """

        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])

        return self.scoreing(y_test, y_pred)
