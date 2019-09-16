"""
utility to plot data
"""

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_regions(
    X, y,
    classifier,
    test_index=None,
    resolution=0.02,
    xlabel='x', ylabel='y',
    loc='best'):
    """
    plot decision regions


    # Parameters
    -----
    * X : 2-d array-like
        training data
    * y : 1-d array-like
        target variable
    * classifier : object
        instance of classifier, which needs to implement the following methods
            classifier.predict : returns the prediction value of a sample
    * resolution : float
        resolution of plot area
    * xlabel : string
        label of x-axis
    * ylabel : string
        label of y-axis
    * loc : string or int
        location of legend
    """

    # prepare for markers and color maps
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    color_map = ListedColormap(colors[:len(np.unique(y))])

    # prepare for plot area and generate grid points
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # predict each grid points
    z = classifier.predict(np.array([x1.ravel(), x2.ravel()]).T)
    z = z.reshape(x1.shape)

    # plot the contour
    plt.contourf(x1, x2, z, alpha=0.3, cmap=color_map)

    # plot training data
    for index, label in enumerate(np.unique(y)):
        plt.scatter(x=X[y == label, 0], y=X[y == label, 1],
                    alpha=0.8, c=colors[index], marker=markers[index], label=label, edgecolor='black')

    # plot test data
    if test_index:
        plt.scatter(x=X[test_idx, 0], y=X[test_idx, 1],
                    alpha=1.0, linewidth=1, marker='o', s=100, label='test_set')

    # configure plot area
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    plt.show()
