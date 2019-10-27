"""
show sample of how to use BaggingClassifier
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_utility import WineData
from plot_utility import plot_decision_regions


def main():
    # prepare sample data and target variable
    features = ['alcohol', 'od280/od315_of_diluted_wines']
    labels = ['class_1', 'class_2']
    wine_data = WineData(features, labels)
    X = wine_data.X
    y = wine_data.y

    # split sample data into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    # fit classifiers
    decision_tree = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=None,
        random_state=1)
    bagging = BaggingClassifier(
        base_estimator=decision_tree,
        n_estimators=500,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=1,
        random_state=1)
    classifiers = [decision_tree, bagging]
    for classifier in classifiers:
        classifier.fit(X_train, y_train)

    names = ['decision tree', 'bagging']
    for classifier, name in zip(classifiers, names):
        # show score
        print('[{name}]'.format(name=name))
        print('training score:', classifier.score(X_train, y_train))
        print('test score:', classifier.score(X_test, y_test))

        # show decision regions
        plot_decision_regions(X_combined, y_combined, classifier=classifier, test_idx=list(range(len(y_train), len(y))), title=name)


if __name__ == '__main__':
    main()
