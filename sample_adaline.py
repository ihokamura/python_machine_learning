"""
show sample of how to use AdalineGD and AdalineSGD
"""

import matplotlib.pyplot as plt
import numpy as np

from adaline import AdalineGD, AdalineSGD
from data_utility import IrisData
from plot_utility import plot_decision_regions


def plot_update_history(classifier):
    costs = classifier.costs_
    plt.title('eta = {}'.format(classifier.eta))
    plt.xlabel('epochs')
    plt.ylabel('log(sum squared error)')
    plt.plot(range(1, len(costs) + 1), np.log10(costs), marker='o')
    plt.show()


def main():
    # prepare training data and target variable
    features = ['sepal length', 'petal length']
    labels = ['setosa', 'versicolor']
    D = IrisData(features, labels)
    X = D.X
    y = np.where(D.y == 0, -1, 1)

    # standardize training data
    X_std = np.copy(X)
    for i in range(len(labels)):
        X_std[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()

    # fit classifiers
    classifiers = [
        AdalineGD(eta=0.01, n_iter=10).fit(X, y),
        AdalineGD(eta=0.0001, n_iter=10).fit(X, y),
        AdalineGD(eta=0.01, n_iter=15).fit(X_std, y),
        AdalineSGD(eta=0.01, n_iter=15).fit(X_std, y)]

    # show history of costs
    for classifier in classifiers:
        plot_update_history(classifier)

    # show decision regions
    plot_decision_regions(
        X_std, y,
        classifier=classifiers[2],
        xlabel='sepal length [standardized]', ylabel='petal lnegth [standardized]')
    plot_decision_regions(
        X_std, y,
        classifier=classifiers[3],
        xlabel='sepal length [standardized]', ylabel='petal lnegth [standardized]')


if __name__ == '__main__':
    main()
