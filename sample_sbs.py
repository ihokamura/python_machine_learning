"""
show sample of how to use SBS
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_utility import WineData
from sbs import SBS


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

    # fit estimators
    estimators = [
        LogisticRegression(C=100.0, random_state=1, solver='liblinear', multi_class='ovr'),
        SVC(C=1.0, kernel='linear', random_state=1).fit(X_train_std, y_train),
        KNeighborsClassifier(n_neighbors=5)
    ]
    sbs_estimators = [SBS(estimator=estimator, k_features=1).fit(X_train_std, y_train) for estimator in estimators]


    # plot score at each steps
    labels = ['logistic regression', 'SVM', 'KNN']
    for sbs, label in zip(sbs_estimators, labels):
        k_features = list(len(subset) for subset in sbs.subsets_)
        plt.plot(k_features, sbs.scores_, marker='o', label=label)
    plt.xlabel('number of feateres')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.show()

    # show results of SBS
    print('[score summary]')
    for sbs, estimator, label in zip(sbs_estimators, estimators, labels):
        print('estimator:', label)

        # search minimal subsets of features which achieves the best score
        indices = sbs.subsets_[0]
        for i in reversed(range(X.shape[1])):
            if sbs.scores_[i] == 1.0:
                indices = sbs.subsets_[i]
                break
        print('minimal subsets:', [wine_data.features[i] for i in indices])

        # compare score with all the features and one with minimal subsets
        estimator_all = estimator.fit(X_train_std, y_train)
        score_all = estimator_all.score(X_test_std, y_test)
        estimator_min = estimator.fit(X_train_std[:, indices], y_train)
        score_min = estimator_min.score(X_test_std[:, indices], y_test)
        print('score (all features)    :', score_all)
        print('score (minimal features):', score_min)


if __name__ == '__main__':
    main()
