"""
show sample of how to compute evaluation metrics with scikit-learn
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import auc, confusion_matrix, f1_score, make_scorer, precision_score, recall_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_utility import BreastCancerData


def main():
    show_confusion_matrix()
    show_evaluation_scores()
    show_roc_curve()


def show_confusion_matrix():
    # prepare sample data and target variable
    labels = None
    features = None
    D = BreastCancerData(features, labels)
    X, y = D.X, D.y

    # split sample data into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # make and fit pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(random_state=1))
    pipeline.fit(X_train, y_train)

    # compute confusion matrix
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # visualize confusion matrix
    _, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(x=i, y=j, s=cm[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.tight_layout()
    plt.show()


def show_evaluation_scores():
    # prepare sample data and target variable
    labels = None
    features = None
    D = BreastCancerData(features, labels)
    X, y = D.X, D.y

    # split sample data into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # make and fit pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(random_state=1))
    pipeline.fit(X_train, y_train)

    # compute evaluation scores
    y_pred = pipeline.predict(X_test)
    print('precision score:', precision_score(y_test, y_pred))
    print('recall score:', recall_score(y_test, y_pred))
    print('f1 score:', f1_score(y_test, y_pred))

    # execute grid search with a custom evaluation score
    scorer = make_scorer(f1_score)
    param_range = list(10**n for n in range(-4, 4))
    param_grid = [
        {'svc__C':param_range, 'svc__kernel':['linear']},
        {'svc__C':param_range, 'svc__gamma':param_range, 'svc__kernel':['rbf']}]
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer, cv=10)
    grid_search = grid_search.fit(X_train, y_train)
    print('best score:', grid_search.best_score_)
    print('best parameters:', grid_search.best_params_)


def show_roc_curve():
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
        LogisticRegression(solver='liblinear', C=100, random_state=1))

    # extract features for ROC curve
    X_train_extracted = X_train[:, [4, 14]]

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr_list = []
    kfold = StratifiedKFold(n_splits=3, random_state=1)
    for i, (idx_train, _) in enumerate(kfold.split(X_train, y_train), start=1):
        # compute fpr (false positive rate) and tpr (true positive rate)
        probas = pipeline.fit(X_train_extracted[idx_train], y_train[idx_train]).predict_proba(X_train_extracted[idx_train])
        fpr, tpr, _ = roc_curve(y_train[idx_train], probas[:, 1], pos_label=1)
        # save interpolation of tpr at fpr in order to compute mean tpr
        mean_tpr_list.append(scipy.interp(mean_fpr, fpr, tpr))
        # plot ROC curve of the current training datasets
        plt.plot(fpr, tpr, label='ROC fold {0} (AUC = {1:f})'.format(i, auc(fpr, tpr)))
    # plot mean ROC curve
    mean_tpr = np.mean(mean_tpr_list, axis=0)
    mean_tpr[0], mean_tpr[-1] = 0, 1
    plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (AUC = {0:f})'.format(auc(mean_fpr, mean_tpr)))
    # plot random guess and perfect estimator
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black')
    # set plot area and show all the plots
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
