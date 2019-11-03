"""
show sample of how to use LinearRegression
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_utility import HousingData
from plot_utility import plot_predictions, plot_residuals


def main():
    show_prediction()
    show_residual()
    show_metrics()


def show_prediction():
    # prepare training data and target variable
    features = ['RM']
    D = HousingData(features)
    X, y = D.X, D.y

    # fit regressor
    regressor = LinearRegression().fit(X, y)

    # show prediction
    plot_predictions(
        X.flatten(), y, regressor,
        xlabel='RM', ylabel='MEDV')

    # show a sample of non-standardized prediction and weights
    rm = [[5.0]]
    medv = regressor.predict(rm)[0]
    print('RM = 5.0 -> MEDV = {:.3e}'.format(medv))

    # show weights
    print('intercept = {i:.3e}, slope = {s:.3e}'.format(i=regressor.intercept_, s=regressor.coef_[0]))


def show_residual():
    # prepare training data and target variable
    features = None
    D = HousingData(features)
    X, y = D.X, D.y

    # split data into training dataset and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    # fit regressor
    regressor = LinearRegression().fit(X_train, y_train)

    # show residual
    plot_residuals(
        X_combined, y_combined,
        regressor, test_idx=range(len(y_train), len(y)),
        xlabel='RM', ylabel='MEDV')


def show_metrics():
    # prepare training data and target variable
    features = None
    D = HousingData(features)
    X, y = D.X, D.y

    # split data into training dataset and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # fit regressor and predict data
    regressor = LinearRegression().fit(X_train, y_train)
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
