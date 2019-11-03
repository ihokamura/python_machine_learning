"""
show sample of how to use LinearRegression
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from data_utility import HousingData
from plot_utility import plot_predictions


def main():
    # prepare training data and target variable
    features = ['RM']
    D = HousingData(features)
    X, y = D.X, D.y

    # fit regressors
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


if __name__ == '__main__':
    main()
