"""
utility to load sample data
"""

import abc

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine


class SampleData(abc.ABC):
    """
    sample dataset in scikit-learn

    # Parameters
    -----
    * features : iterable
        list of features in training data
        All the features are retrieved if None.
    * labels : iterable
        list of labels for target variable
        All the labels are retrieved if None.

    # Attributes
    -----
    * X : array-like, shape = ((number of samples), (number of features))
        training data
    * y : array-like, shape = ((number of samples), )
        target variable
    * features : list
        list of features in training data
    * labels : list
        list of labels for target variable
    """

    def __init__(self, features=None, labels=None):
        data_bunch = self.load_data()
        if features is None:
            features = data_bunch.feature_names
        feature_index = np.array(list(f in features for f in data_bunch.feature_names))
        if labels is None:
            labels = data_bunch.target_names
        label_map = {value:name for value, name in enumerate(data_bunch.target_names)}
        label_index = np.array(list(label_map[t] in labels for t in data_bunch.target))

        X = data_bunch.data[label_index][:, feature_index]
        y = data_bunch.target[label_index]

        self.__features = list(features)
        self.__labels = list(labels)
        self.__X = X
        self.__y = y

    @property
    def X(self):
        return self.__X

    @property
    def y(self):
        return self.__y

    @property
    def features(self):
        return self.__features

    @property
    def labels(self):
        return self.__labels

    @abc.abstractclassmethod
    def load_data(cls):
        """
        load the original data
        """


class IrisData(SampleData):
    """
    Iris dataset

    # Parameters
    -----
    * features : iterable
        list of features in training data
        All the features are retrieved if None.
        The Iris dataset has the following features.
            * sepal length (cm)
            * sepal width (cm)
            * petal length (cm)
            * petal width (cm)
    * labels : iterable
        list of labels for target variable
        All the labels are retrieved if None.
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
    * features : list
        list of features in training data
    * labels : list
        list of labels for target variable
    """

    @classmethod
    def load_data(cls):
        return load_iris()


class WineData(SampleData):
    """
    Wine dataset

    # Parameters
    -----
    * features : iterable
        list of features in training data
        All the features are retrieved if None.
        The Wine dataset has the following features.
            * alcohol
            * malic_acid
            * ash
            * alcalinity_of_ash
            * magnesium
            * total_phenols
            * flavanoids
            * nonflavanoid_phenols
            * proanthocyanins
            * color_intensity
            * hue
            * od280/od315_of_diluted_wines
            * proline
    * labels : iterable
        list of labels for target variable
        All the labels are retrieved if None.
        The Wine dataset has the following labels.
            * class_0
            * class_1
            * class_2

    # Attributes
    -----
    * X : array-like, shape = ((number of samples), (number of features))
        training data
    * y : array-like, shape = ((number of samples), )
        target variable
    * features : list
        list of features in training data
    * labels : list
        list of labels for target variable
    """

    @classmethod
    def load_data(cls):
        return load_wine()


class BreastCancerData(SampleData):
    """
    Breast Cancer Wisconsin dataset

    # Parameters
    -----
    * features : iterable
        list of features in training data
        All the features are retrieved if None.
        The Breast Cancer Wisconsin dataset has the following features.
            * mean radius
            * mean texture
            * mean perimeter
            * mean area
            * mean smoothness
            * mean compactness
            * mean concavity
            * mean concave points
            * mean symmetry
            * mean fractal dimension
            * radius error
            * texture error
            * perimeter error
            * area error
            * smoothness error
            * compactness error
            * concavity error
            * concave points error
            * symmetry error
            * fractal dimension error
            * worst radius
            * worst texture
            * worst perimeter
            * worst area
            * worst smoothness
            * worst compactness
            * worst concavity
            * worst concave points
            * worst symmetry
            * worst fractal dimension
    * labels : iterable
        list of labels for target variable
        All the labels are retrieved if None.
        The Breast Cancer Wisconsin dataset has the following labels.
            * malignant
            * benign

    # Attributes
    -----
    * X : array-like, shape = ((number of samples), (number of features))
        training data
    * y : array-like, shape = ((number of samples), )
        target variable
    * features : list
        list of features in training data
    * labels : list
        list of labels for target variable
    """

    @classmethod
    def load_data(cls):
        return load_breast_cancer()
