"""
show sample of how to use HousingData
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np

from data_utility import HousingData


def plot_housing_data():
    features = ['LSTAT', 'INDUS', 'NOX', 'RM']
    D = HousingData(features)
    variables = np.hstack((D.X, D.y[:, np.newaxis])).T
    labels = features + ['MEDV']

    # visualize correlation matrix
    correlation = np.corrcoef(variables)
    _, ax = plt.subplots()
    ax.matshow(correlation, cmap=plt.cm.Blues, alpha=0.3)
    for i, j in itertools.product(range(correlation.shape[0]), range(correlation.shape[1])):
        ax.text(x=i, y=j, s='{:.2f}'.format(correlation[i, j]), va='center', ha='center')
    plt.tight_layout()
    plt.show()

    # show scatter plot and histogram
    nrows = ncols = variables.shape[0]
    _, ax = plt.subplots(nrows, ncols)
    for i in range(nrows): # iteration from top to bottom in y-axis
        for j in range(ncols): # iteration from left to right in x-axis
            if i == j:
                ax[i, j].hist(variables[i])
            else:
                ax[i, j].scatter(x=variables[j], y=variables[i], edgecolors='white')

            if i == nrows - 1:
                ax[i, j].set_xlabel(labels[j])
            if j == 0:
                ax[i, j].set_ylabel(labels[i])
    plt.show()


if __name__ == '__main__':
    plot_housing_data()
