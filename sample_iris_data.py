"""
show sample of how to use IrisData
"""

import matplotlib.pyplot as plt

from data_utility import IrisData


def plot_iris_data():
    features = ['sepal length (cm)', 'petal length (cm)']
    labels = ['setosa', 'versicolor', 'virginica']
    colors = ['red', 'blue', 'green']
    markers = ['o', 'x', '^']

    for label, color, marker in zip(labels, colors, markers):
        X = IrisData(features, [label]).X
        plt.scatter(X[:, 0], X[:, 1], color=color, marker=marker, label=label)

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    plot_iris_data()
