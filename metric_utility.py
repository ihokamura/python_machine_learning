"""
utility to compute metrics
"""

import numpy as np


def roc_curve(y_true, y_score, pos_label):
    """
    compute ROC (receiver operating characteristic) curve

    # Parameters
    -----
    * y_true : array-like, shape = (n_samples, )
        target variable
    * y : array-like, shape = (n_samples, )
        score (model prediction) corresponding to the target variable
    * pos_label : int
        label of target variable seen as positive

    # Returns
    -----
    * fpr : array-like, shape = (n_samples, )
        false positive rate at each thresholds
    * tpr : array-like, shape = (n_samples, )
        true positive rate at each thresholds
    * threshold : array-like, shape = (n_samples, )
        thresholds to compute ROC curve in descending order

    # Notes
    -----
    * n_samples represents the number of samples.
    """

    # determine negative label
    if y_true[0] == pos_label:
        neg_label = y_true[1]
    else:
        neg_label = y_true[0]

    fpr, tpr = [], []
    thresholds = np.array(sorted(y_score, reverse=True))
    for threshold in thresholds:
        # count true positive, true negative, false positive and false negative
        N_tp = np.sum([(y == pos_label and score >= threshold) for y, score in zip(y_true, y_score)])
        N_tn = np.sum([(y == neg_label and score <  threshold) for y, score in zip(y_true, y_score)])
        N_fp = np.sum([(y == neg_label and score >= threshold) for y, score in zip(y_true, y_score)])
        N_fn = np.sum([(y == pos_label and score <  threshold) for y, score in zip(y_true, y_score)])
        fpr.append(N_fp/(N_tn + N_fp))
        tpr.append(N_tp/(N_tp + N_fn))
    fpr = np.array(fpr)
    tpr = np.array(tpr)

    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """
    compute AUC (area under ROC curve)

    # Parameters
    -----
    * fpr : array-like, shape = (n_samples, )
        false positive rate at each thresholds
    * tpr : array-like, shape = (n_samples, )
        true positive rate at each thresholds

    # Returns
    -----
    * _ : float
        AUC for the ROC

    # Notes
    -----
    * n_samples represents the number of samples.
    """

    return np.trapz(tpr, fpr)
