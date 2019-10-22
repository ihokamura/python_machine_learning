"""
show sample of how to use auc and roc_curve
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_utility import BreastCancerData
import metric_utility


def main():
    # prepare sample data and target variable
    labels = None
    features = None
    D = BreastCancerData(features, labels)
    X, y = D.X, D.y

    # split sample data into training data and test data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # make and pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression(solver='liblinear', C=100, random_state=1))

    # extract features and compute scores for ROC curve
    X_train_extracted = X_train[:, [4, 14]]
    probas = pipeline.fit(X_train_extracted, y_train).predict_proba(X_train_extracted)

    roc_functions = (metric_utility.roc_curve, roc_curve)
    auc_functions = (metric_utility.auc, auc)

    for i, (roc_func, auc_func) in enumerate(zip(roc_functions, auc_functions), start=1):
        # compute fpr (false positive rate) and tpr (true positive rate)
        fpr, tpr, _ = roc_func(y_train, probas[:, 1], pos_label=1)
        # plot ROC curve
        plt.plot(fpr, tpr, label='ROC {0} (AUC = {1:f})'.format(i, auc_func(fpr, tpr)))

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
