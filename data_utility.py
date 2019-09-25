"""
utility to load sample data
"""

import numpy as np
from sklearn.datasets import load_iris


class IrisData:
    """
    Iris dataset

    # Parameters
    -----
    * features : iterable
        list of features in training data
        The Iris dataset has the following features.
            * sepal length (cm)
            * sepal width (cm)
            * petal length (cm)
            * petal width (cm)
    * labels : iterable
        list of labels for target variable
        The Iris dataset has the following labels.
            * setosa
            * versicolor
            * virginica

    # Attributes
    -----
    * X : array-like, shape = ((number of samples), (number of features))
        training data
    * y : array-like, shape = ((number of samples), )
        target variable
    """

    def __init__(self, features, labels):
        data_bunch = load_iris()
        feature_index = np.array(list(f in features for f in data_bunch.feature_names))
        label_map = {value:name for value, name in enumerate(data_bunch.target_names)}
        label_index = np.array(list(label_map[t] in labels for t in data_bunch.target))

        X = data_bunch.data[label_index][:, feature_index]
        y = data_bunch.target[label_index]

        self.__X = X
        self.__y = y

    @property
    def X(self):
        return self.__X

    @property
    def y(self):
        return self.__y
