"""
show sample of how to use MajorityVoteClassifier
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_utility import IrisData
from majority_vote import MajorityVoteClassifier
from plot_utility import plot_decision_regions


def main():
    # prepare sample data and target variable
    labels = ['versicolor', 'virginica']
    features = ['sepal width (cm)', 'petal length (cm)']
    D = IrisData(features, labels)
    X = D.X
    y = D.y

    # split sample data into training data and test data and standardize them
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)
    sc = StandardScaler().fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # combine training data and test data
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # prepare classifiers
    logistic_regression = LogisticRegression(penalty='l2', solver='liblinear', C=0.001, random_state=1)
    decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    majority_vote = MajorityVoteClassifier(classifiers=[logistic_regression, decision_tree, knn])
    classifiers = [logistic_regression, decision_tree, knn, majority_vote]
    classifier_names = ['logistic regression', 'decision tree', 'KNN', 'majority vote']

    # compute cross validation score of classifiers
    for classifier, name in zip(classifiers, classifier_names):
        scores = cross_val_score(
            estimator=classifier,
            X=X_train_std, y=y_train,
            cv=10, scoring='accuracy')
        print('accuracy : {mean:f} +/- {std:f} ({name})'.format(mean=np.mean(scores), std=np.std(scores), name=name))

    # execute grid search
    param_grid = {
        'decisiontreeclassifier__max_depth':[1, 2],
        'logisticregression__C':[0.001, 0.1, 100.0]}
    grid = GridSearchCV(estimator=majority_vote, param_grid=param_grid, cv=10, scoring='accuracy').fit(X_train_std, y_train)
    for mean, std, params in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['std_test_score'], grid.cv_results_['params']):
        print('accuracy: {mean:f} +/- {std:f} with {params}'.format(mean=mean, std=std, params=params))

    # plot ROC curves
    pos_label_index = 1
    for classifier, name in zip(classifiers, classifier_names):
        y_pred = classifier.fit(X_train_std, y_train).predict_proba(X_test_std)[:, pos_label_index]
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=pos_label_index + 1)
        plt.plot(fpr, tpr, label='{name} (AUC = {auc:f})'.format(name=name, auc=auc(fpr, tpr)))
    plt.grid(alpha=0.5)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(loc='lower right')
    plt.show()

    # plot decision regions
    for classifier in classifiers:
        classifier.fit(X_train_std, y_train)
        plot_decision_regions(X_combined_std, y_combined, classifier=classifier, test_idx=list(range(len(y_train), len(y))))


if __name__ == '__main__':
    main()
