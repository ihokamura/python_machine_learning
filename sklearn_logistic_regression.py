"""
show sample of how to use LogisticRegression
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_utility import IrisData
from plot_utility import plot_decision_regions


def main():
    # prepare sample data and target variable
    labels = ['setosa', 'versicolor', 'virginica']
    features = ['petal length (cm)', 'petal width (cm)']
    D = IrisData(features, labels)
    X = D.X
    y = D.y

    # split sample data into training data and test data and standardize them
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler().fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # combine training data and test data
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # fit classifiers
    classifier = LogisticRegression(C=100.0, random_state=1, solver='liblinear', multi_class='ovr').fit(X_train_std, y_train)

    # show accuracy
    y_pred = classifier.predict(X_test_std)
    print('misclassified samples: {}'.format(np.sum(y_test != y_pred)))

    # show decision regions
    plot_decision_regions(X_combined_std, y_combined, classifier=classifier, test_idx=list(range(105, 150)))

    # show effect of regularization parameter
    weights = []
    params = []
    for i in np.arange(-5, 5):
        C = 10.0**i
        classifier = LogisticRegression(C=C, random_state=1, solver='liblinear', multi_class='ovr').fit(X_train_std, y_train)
        weights.append(classifier.coef_[1])
        params.append(C)
    weights = np.array(weights)
    plt.plot(params, weights[:, 0], label='petal length')
    plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('weight coefficient')
    plt.show()


if __name__ == '__main__':
    main()
