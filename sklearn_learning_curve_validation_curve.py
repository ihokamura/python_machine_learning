"""
show sample of how to use learning_curve and validation_curve
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
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
    
    # show learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=pipeline,
        X=X_train, y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=10,
        random_state=1)
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='training accuracy', marker='o')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.25)
    plt.plot(train_sizes, test_mean, label='test accuracy', marker='o')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.25)
    plt.grid()
    plt.ylim(top=1.0)
    plt.title('learning curve')
    plt.legend(loc='upper right')
    plt.xlabel('number of training samples')
    plt.ylabel('accuracy')
    plt.show()

    # show validation curve
    params = [10**n for n in range(-3   , 3)]
    train_scores, test_scores = validation_curve(
        estimator=pipeline,
        X=X_train, y=y_train,
        param_name='logisticregression__C',
        param_range=params,
        cv=10)
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)
    plt.plot(params, train_mean, label='training accuracy', marker='o')
    plt.fill_between(params, train_mean + train_std, train_mean - train_std, alpha=0.25)
    plt.plot(params, test_mean, label='test accuracy', marker='o')
    plt.fill_between(params, test_mean + test_std, test_mean - test_std, alpha=0.25)
    plt.grid()
    plt.xscale('log')
    plt.ylim(top=1.0)
    plt.title('validation curve')
    plt.legend(loc='upper right')
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.show()

    # compute the final score
    C = params[np.argmax(test_mean)]
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression(C=C, solver='liblinear', random_state=1))
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print('final score:', score)


if __name__ == '__main__':
    main()
