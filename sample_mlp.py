"""
show sample of how to use MLP
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlp import MLP

IMAGE_SHAPE = (8, 8)


def show_images(X, y, y_pred):
        _, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
        ax = ax.flatten()
        for axis, data, label, label_pred in zip(ax, X, y, y_pred):
            axis.imshow(data.reshape(IMAGE_SHAPE), cmap='Greys')
            axis.set_title('true:{0}/pred:{1}'.format(label, label_pred))
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()


def show_learning_history(classifier):
    history = [classifier.costs_, classifier.accuracys_]
    ylabels = ['cost', 'accuracy']

    _, ax = plt.subplots(ncols=len(history))
    for axis, values, ylabel in zip(ax, history, ylabels):
        axis.set_xlabel('epochs')
        axis.set_ylabel(ylabel)
        axis.plot(range(1, len(values) + 1), values)
    plt.tight_layout()
    plt.show()


def main():
    # prepare sample data and target variable
    X, y = load_digits(return_X_y=True)

    # split sample data into training data and test data and standardize them
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler().fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # compare performance of MLP classifiers with different parameters
    classifiers = [
        MLP(n_hidden=100, l2=0.01, epochs=200, eta=0.0005, minibatch_size=100, shuffle=True, seed=1),
        MLP(n_hidden=100, l2=0.01, epochs=200, eta=0.01, minibatch_size=100, shuffle=True, seed=1),
        MLP(n_hidden=100, l2=1.0, epochs=200, eta=0.0005, minibatch_size=100, shuffle=True, seed=1),
        MLP(n_hidden=10, l2=0.01, epochs=200, eta=0.0005, minibatch_size=100, shuffle=True, seed=1)
        ]
    for classifier in classifiers:
        # fit classifier
        classifier.fit(X_train_std, y_train)

        # show accuracy
        y_pred = classifier.predict(X_test_std)
        print('test accuracy: {}'.format(accuracy_score(y_test, y_pred)))

        # show some misclassified images
        indices = (y_test != y_pred)
        show_images(X_test[indices], y_test[indices], y_pred[indices])

        # show learning history
        show_learning_history(classifier)


if __name__ == '__main__':
    main()
