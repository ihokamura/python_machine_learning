"""
show sample of how to use RandomForestClassifier
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from data_utility import IrisData
from plot_utility import plot_decision_regions


def main():
    # prepare sample data and target variable
    labels = ['setosa', 'versicolor', 'virginica']
    features = ['petal length (cm)', 'petal width (cm)']
    D = IrisData(features, labels)
    X = D.X
    y = D.y

    # split sample data into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # combine training data and test data
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    # fit classifiers
    classifiers = [
        RandomForestClassifier(criterion='gini', n_estimators=10, random_state=1, n_jobs=2).fit(X_train, y_train),
        RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2).fit(X_train, y_train),
        RandomForestClassifier(criterion='gini', n_estimators=50, random_state=1, n_jobs=2).fit(X_train, y_train),
        RandomForestClassifier(criterion='gini', n_estimators=100, random_state=1, n_jobs=2).fit(X_train, y_train),
        RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=1, n_jobs=2).fit(X_train, y_train)
    ]

    for classifier in classifiers:
        # show accuracy
        y_pred = classifier.predict(X_test)
        print('misclassified samples: {}'.format(np.sum(y_test != y_pred)))

        # show decision regions
        plot_decision_regions(X_combined, y_combined, classifier=classifier, test_idx=list(range(len(y_train), len(y_test))))


if __name__ == '__main__':
    main()
