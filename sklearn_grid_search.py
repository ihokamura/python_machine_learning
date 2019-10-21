"""
show sample of how to use GridSearchCV
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_utility import BreastCancerData


def main():
    # prepare sample data and target variable
    labels = None
    features = None
    D = BreastCancerData(features, labels)
    X, y = D.X, D.y

    # split sample data into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # make pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(random_state=1))

    # execute grid search
    param_range = list(10**n for n in range(-4, 4))
    param_grid = [
        {'svc__C':param_range, 'svc__kernel':['linear']},
        {'svc__C':param_range, 'svc__gamma':param_range, 'svc__kernel':['rbf']}]
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy', cv=10)
    grid_search = grid_search.fit(X_train, y_train)
    print('best score:', grid_search.best_score_)
    print('best parameters:', grid_search.best_params_)

    # compute the final score
    estimator = grid_search.best_estimator_.fit(X_train, y_train)
    score = estimator.score(X_test, y_test)
    print('final score:', score)

    # execute nested cross validation (5x2 cross validation)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy', cv=2)
    scores = cross_val_score(grid_search, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy (SVM): {0:f} +/- {1:f}'.format(np.mean(scores), np.std(scores)))
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=1),
        param_grid=[{'max_depth':[1, 2, 3, 4, 5, 6, 7, None]}],
        scoring='accuracy', cv=2)
    scores = cross_val_score(grid_search, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy (decision tree): {0:f} +/- {1:f}'.format(np.mean(scores), np.std(scores)))


if __name__ == '__main__':
    main()
