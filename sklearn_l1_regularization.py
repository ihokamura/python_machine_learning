"""
show effect of L1 regularization
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_utility import WineData


def main():
    # prepare sample data and target variable
    wine_data = WineData()
    X = wine_data.X
    y = wine_data.y

    # split sample data into training data and test data and standardize them
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler().fit(X_train)
    X_train_std = sc.transform(X_train)                                                                         
    X_test_std = sc.transform(X_test)

    # fit classifiers
    classifier = LogisticRegression(penalty='l1', C=100.0, random_state=1, solver='liblinear', multi_class='ovr').fit(X_train_std, y_train)

    # show score
    print('training accuracy:', classifier.score(X_train_std, y_train))
    print('test accuracy:', classifier.score(X_test_std, y_test))

    # show effect of regularization parameter
    weights = []
    params = []
    for i in np.arange(-4, 6):
        C = 10.0**i
        classifier = LogisticRegression(penalty='l1', C=C, random_state=1, solver='liblinear', multi_class='ovr').fit(X_train_std, y_train)
        weights.append(classifier.coef_[1])
        params.append(C)
    weights = np.array(weights)
    for i, label in enumerate(wine_data.features):
        plt.plot(params, weights[:, i], label=label)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10**-5, 10**5])
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('weight coefficient')
    plt.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':
    main()
