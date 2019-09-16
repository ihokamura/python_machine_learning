"""
utility to load sample data
"""

import numpy as np
import pandas as pd


class IrisData:
    """
    Iris dataset

    # Parameters
    -----
    * features : iterable
        list of features in training data
        The Iris dataset has the following features.
            * sepal length
            * sepal width
            * petal length
            * petal width
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

    PATH = '/usr/local/lib/python3.7/site-packages/sklearn/datasets/data/iris.csv'
    DF = pd.read_csv(PATH)

    LABEL_RANGE_MAP = {
        'setosa':slice(0, 50),
        'versicolor':slice(50, 100),
        'virginica':slice(100, 150)}
    FEATURE_INDEX_MAP = {
        'sepal length':0,
        'sepal width':1,
        'petal length':2,
        'petal width':3,
        'label':4
    }


    def __init__(self, features, labels):
        feature_indexes = list(self.FEATURE_INDEX_MAP[f] for f in features)
        data_stack = list(self.DF.iloc[self.LABEL_RANGE_MAP[l], feature_indexes].values for l in labels)
        X = np.vstack(data_stack)

        label_stack = list(self.DF.iloc[self.LABEL_RANGE_MAP[l], self.FEATURE_INDEX_MAP['label']].values for l in labels)
        y = np.vstack(label_stack).ravel()

        self.__X = X
        self.__y = y

    @property
    def X(self):
        return self.__X

    @property
    def y(self):
        return self.__y
