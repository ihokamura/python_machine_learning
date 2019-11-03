"""
show sample of how to use LinearRegressionGD and LinearRegressionSGD
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_utility import HousingData
from linear_regression import LinearRegressionGD, LinearRegressionSGD
from plot_utility import plot_predictions


def plot_update_history(regressor):
    costs = regressor.costs_
    plt.title(r'$\eta$ = {}'.format(regressor.eta))
    plt.xlabel('epochs')
    plt.ylabel('log(sum squared error)')
    plt.plot(range(1, len(costs) + 1), np.log10(costs), marker='o')
    plt.show()


def main():
    # prepare training data and target variable
    features = ['RM']
    D = HousingData(features)
    X, y = D.X, D.y

    # standardize training data
    sc_x = StandardScaler().fit(X)
    sc_y = StandardScaler().fit(y[:, np.newaxis])
    X_std = sc_x.transform(X)
    y_std = sc_y.transform(y[:, np.newaxis]).flatten()

    # fit regressors
    regressors = [
        LinearRegressionGD(eta=0.01, n_iter=20).fit(X_std, y_std),
        LinearRegressionGD(eta=0.001, n_iter=20).fit(X_std, y_std),
        LinearRegressionSGD(eta=0.01, n_iter=100).fit(X_std, y_std),
    ]

    for regressor in regressors:
        # show history of costs
        plot_update_history(regressor)

        # show prediction
        plot_predictions(
            X_std, y_std, regressor,
            xlabel='RM (standardized)', ylabel='MEDV (standardized)', title=r'$\eta$ = {}'.format(regressor.eta))

        # show a sample of non-standardized prediction and weights
        rm_std = sc_x.transform([[5.0]])
        medv_std = regressor.predict(rm_std)
        medv = sc_y.inverse_transform(medv_std)[0]
        print('RM = 5.0 -> MEDV = {:.3e}'.format(medv))

        # show weights
        print('intercept = {i:.3e}, slope = {s:.3e}'.format(i=regressor.w_[0], s=regressor.w_[1]))


if __name__ == '__main__':
    main()
