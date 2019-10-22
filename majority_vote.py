"""
majority vote ensemble classifier
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import LabelEncoder


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """
    majority vote ensemble classifier

    # Parameters
    -----
    * classifiers : array-like, shape = (n_classifiers, )
        ensemble classifiers
    * vote : str
        policy of majority vote
        One of the following must be specified:
            * 'classlabel' : predict based on argmax of class labels
            * 'probability' : predict based on argmax of probabilities
    * weights : array-like, shape = (n_classifiers, )
        weights of classifiers used for majority vote
        Uniform weights are used if None.

    # Notes
    -----
    * n_classifiers represents the number of ensemble classifiers.
    * n_samples represents the number of samples.
    * n_features represents the number of features.
    """

    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self._names = _name_estimators(classifiers)
        self.vote = vote
        self.weights = weights

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
        * self : MajorityVoteClassifier
            majority vote ensemble classifier after fit
        """

        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        self.classifiers_ = list(clone(classifier).fit(X, self.label_encoder_.transform(y)) for classifier in self.classifiers)

        return self

    def predict(self, X):
        """
        predict class label

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
             data of which label is predicted

        # Returns
        -----
        * _ : array-like, shape = (n_samples, )
            predicted class labels
        """

        if self.vote == 'probability':
            majority_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray(list(classifier.predict(X) for classifier in self.classifiers_)).T
            majority_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions)

        return self.label_encoder_.inverse_transform(majority_vote)

    def predict_proba(self, X):
        """
        predict probabilities

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
             data of which probabilities are predicted

        # Returns
        -----
        * _ : array-like, shape = (n_samples, )
            predicted probabilities
        """

        probas = np.array(list(classifier.predict_proba(X) for classifier in self.classifiers))

        return np.average(probas, axis=0, weights=self.weights)

    def get_params(self, deep=True):
        """
        get parameters

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
             data of which probabilities are predicted

        # Returns
        -----
        * _ : dict
            dictionary of parameters
        """

        if deep is not True:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            params = dict(self._names)
            for name, step in self._names:
                for key, value in step.get_params(deep=True).items():
                    params['{name}__{key}'.format(name=name, key=key)] = value

            return params
