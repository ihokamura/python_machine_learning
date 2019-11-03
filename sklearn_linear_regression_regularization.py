"""
show sample of how to use ElasticNet, Lasso and Ridge
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_utility import HousingData
from plot_utility import plot_residuals


def main():
    # prepare training data and target variable
    features = None
    D = HousingData(features)
    X, y = D.X, D.y

    # split data into training dataset and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    # fit regressors
    regressors = [
        Ridge(alpha=1.0).fit(X_train, y_train),
        Lasso(alpha=1.0).fit(X_train, y_train),
        ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X_train, y_train)
    ]
    names = ['ridge', 'LASSO', 'elastic net']

    # show residual
    for regressor, name in zip(regressors, names):
        plot_residuals(
            X_combined, y_combined,
            regressor, test_idx=range(len(y_train), len(y)),
            xlabel='RM', ylabel='MEDV', title=name)

        # show scores of regressor
        print('<{}>'.format(name))
        y_pred_train = regressor.predict(X_train)
        y_pred_test = regressor.predict(X_test)

        # compute mean squared error
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        print('[MSE] train:{0:.3f} / test:{1:.3f}'.format(mse_train, mse_test))

        # compute R^2 score
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        print('[R^2 score] train:{0:.3f} / test:{1:.3f}'.format(r2_train, r2_test))    


if __name__ == '__main__':
    main()
