"""
show sample of how to use Perceptron
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perceptron import Perceptron
from data_utility import IrisData
from plot_utility import plot_decision_regions


def plot_update_history(classifier):
    errors = classifier.errors_
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel('epochs')
    plt.ylabel('number of updates')
    plt.show()


def main():
    # prepare training data and target variable
    features = ['sepal length', 'petal length']
    labels = ['setosa', 'versicolor']
    D = IrisData(features, labels)
    X = D.X
    y = np.where(D.y == 0, -1, 1)

    # fit perceptron
    classifier = Perceptron(eta=0.1, n_iter=10)
    classifier.fit(X, y)

    # show history of errors
    plot_update_history(classifier)

    # show decision regions
    plot_decision_regions(X, y, classifier=classifier, 
                          xlabel='sepal length [cm]', ylabel='petal lnegth [cm]')


if __name__ == '__main__':
    main()
