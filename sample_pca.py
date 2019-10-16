"""
show sample of how to use PCA transformer
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_utility import WineData
import pca
from plot_utility import plot_decision_regions, plot_features


def main():
    # prepare sample data and target variable
    wine_data = WineData()
    X = wine_data.X
    y = wine_data.y

    # split sample data into training data and test data and standardize them
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    pca_transformers = [
        pca.PCA(n_components=2),
        PCA(n_components=2)
    ]
    for pca_transformer in pca_transformers:
        # execute PCA
        X_train_pca = pca_transformer.fit_transform(X_train_std)

        # show principal components and explained variance
        print('principal components:\n', pca_transformer.components_)
        print('explained variance:', pca_transformer.explained_variance_)
        plot_features(X_train_pca, y_train, xlabel='PC1', ylabel='PC2')

        # fit classifier and plot decigion regions
        classifier = LogisticRegression(C=100.0, random_state=1, solver='liblinear', multi_class='ovr').fit(X_train_pca, y_train)
        X_test_pca = pca_transformer.transform(X_test_std)
        print('score: ', classifier.score(X_test_pca, y_test))
        plot_decision_regions(X_test_pca, y_test, classifier=classifier, xlabel='PC1', ylabel='PC2')


if __name__ == '__main__':
    main()
