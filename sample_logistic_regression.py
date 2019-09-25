"""
show sample of how to use LogisticRegressionGD
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logistic_regression import LogisticRegressionGD
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
    # prepare sample data and target variable
    labels = ['setosa', 'versicolor']
    features = ['petal length (cm)', 'petal width (cm)']
    D = IrisData(features, labels)
    X = D.X
    y = D.y

    # split sample data into training data and test data and standardize them
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler().fit(X_train)
    X_train_std = sc.transform(X_train)                                                                         
    X_test_std = sc.transform(X_test)

    # fit classifiers
    classifier = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1).fit(X_train_std, y_train)

    # show accuracy
    y_pred = classifier.predict(X_test_std)
    print('misclassified samples: {}'.format(np.sum(y_test != y_pred)))

    # show history of costs
    plot_update_history(classifier)

    # show decision regions
    plot_decision_regions(X_train_std, y_train, classifier=classifier)


if __name__ == '__main__':
    main()
