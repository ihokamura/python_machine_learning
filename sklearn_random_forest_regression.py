"""
show sample of how to use DecisionTreeRegressor and RandomForestRegressor
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from data_utility import HousingData
from plot_utility import plot_predictions, plot_residuals


def main():
    regress_by_decision_tree()
    regress_by_random_forest()


def regress_by_decision_tree():
    # prepare training data and target variable
    features = ['LSTAT']
    D = HousingData(features)
    X, y = D.X, D.y

    # fit regressors
    regressor = DecisionTreeRegressor(max_depth=3).fit(X, y)

    # plot prediction
    sort_idx = X.flatten().argsort()
    plot_predictions(X[sort_idx].flatten(), y[sort_idx], regressor, xlabel='LSTAT', ylabel='MEDV')


def regress_by_random_forest():
    # prepare training data and target variable
    features = None
    D = HousingData(features)
    X, y = D.X, D.y

    # split data into training dataset and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    # fit regressors
    regressor = RandomForestRegressor(n_estimators=1000, random_state=1, n_jobs=-1).fit(X_train, y_train)

    # compute scores
    scorers = [mean_squared_error, r2_score]
    scorer_names = ['MSE', 'R2']
    for score, scorer_name in zip(scorers, scorer_names):
        print('[{scorer_name}]'.format(scorer_name=scorer_name))
        print('training data:{score:.3f}'.format(score=score(y_train, regressor.predict(X_train))))
        print('test data:{score:.3f}'.format(score=score(y_test, regressor.predict(X_test))))

    # plot residuals
    plot_residuals(X_combined, y_combined, regressor, test_idx=range(len(y_train), len(y)))


if __name__ == '__main__':
    main()
