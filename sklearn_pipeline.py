"""
show sample of how to use PipeLine
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_utility import BreastCancerData
from plot_utility import plot_decision_regions


def main():
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
        PCA(n_components=2),
        LogisticRegression(random_state=1))
    pipeline.fit(X_train, y_train)

    # show accuracy
    y_pred = pipeline.predict(X_test)
    print('misclassified samples: {}'.format(np.sum(y_test != y_pred)))


if __name__ == '__main__':
    main()
