"""
show sample of how to implement cross validation
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_utility import BreastCancerData


def main():
    # prepare sample data and target variable
    labels = None
    features = None
    D = BreastCancerData(features, labels)
    X, y = D.X, D.y

    # split sample data into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # make and pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression(solver='liblinear', random_state=1))

    # explicitly execute stratified k-fold cross validation
    kfold = StratifiedKFold(n_splits=10, random_state=1)
    scores = []
    for k, (idx_train, idx_test) in enumerate(kfold.split(X_train, y_train), start=1):
        pipeline.fit(X_train[idx_train], y_train[idx_train])
        score = pipeline.score(X_train[idx_test], y_train[idx_test])
        scores.append(score)
        print('fold: {0:2d} | class distribution: {1} | score: {2:f}'.format(
            k, np.bincount(y_train[idx_train]) / len(y_train[idx_train]), score))
    print('CV accuracy: {0:f} +/- {1:f}'.format(np.mean(scores), np.std(scores)))

    # use cross_val_score function for cross validation
    scores = cross_val_score(estimator=pipeline, X=X_train, y=y_train, cv=10, n_jobs=1)
    for k, score in enumerate(scores, start=1):
        print('fold: {0:2d} | score: {1:f}'.format(k, score))
    print('CV accuracy: {0:f} +/- {1:f}'.format(np.mean(scores), np.std(scores)))

    # compute the final score
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print('final score:', score)


if __name__ == '__main__':
    main()
