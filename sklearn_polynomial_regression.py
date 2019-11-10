"""
show sample of how to use PolynomialFeatures
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

from data_utility import HousingData


def main():
    regress_artificial_data()
    regress_housing_data()


def regress_artificial_data():
    # prepare training data and target variable
    X = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
    y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])

    # fit regressors
    transformers = [
        PolynomialFeatures(degree=1).fit_transform,
        PolynomialFeatures(degree=2).fit_transform
    ]
    regressors = [LinearRegression().fit(transform(X), y) for transform in transformers]
    regressor_names = ['linear', 'polynomial']

    # compute scores
    scorers = [mean_squared_error, r2_score]
    scorer_names = ['MSE', 'R2']
    for score, scorer_name in zip(scorers, scorer_names):
        print('[{scorer_name}]'.format(scorer_name=scorer_name))
        for transform, regressor, regressor_name in zip(transformers, regressors, regressor_names):
            y_pred = regressor.predict(transform(X))
            print('{regressor_name}:{score:.3f}'.format(regressor_name=regressor_name, score=score(y, y_pred)))

    # plot training data
    plt.scatter(X.flatten(), y, label='training data')
    # plot prediction of regressors
    X_pred = np.arange(250, 600, 10)[:, np.newaxis]
    for transform, regressor, regressor_name in zip(transformers, regressors, regressor_names):
        y_pred = regressor.predict(transform(X_pred))
        plt.plot(X_pred.flatten(), y_pred, label=regressor_name)
    # set plot area
    plt.legend()
    plt.tight_layout()
    plt.show()


def regress_housing_data():
    # prepare training data and target variable
    features = ['LSTAT']
    D = HousingData(features)
    X, y = D.X, D.y

    # fit regressors
    transformers = [
        PolynomialFeatures(degree=1).fit_transform,
        PolynomialFeatures(degree=2).fit_transform,
        PolynomialFeatures(degree=3).fit_transform
    ]
    regressors = [LinearRegression().fit(transform(X), y) for transform in transformers]
    regressor_names = ['linear', 'polynomial (d=2)', 'polynomial (d=3)']

    # compute scores
    print('[R2 score]')
    for transform, regressor, regressor_name in zip(transformers, regressors, regressor_names):
        score = r2_score(y, regressor.predict(transform(X)))
        print('{name}:{score:.3f}'.format(name=regressor_name, score=score))

    # plot training data
    plt.scatter(X.flatten(), y, label='training data', edgecolor='white')
    # plot prediction of regressors
    X_pred = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
    for transform, regressor, regressor_name in zip(transformers, regressors, regressor_names):
        y_pred = regressor.predict(transform(X_pred))
        plt.plot(X_pred.flatten(), y_pred, label=regressor_name)
    # set plot area
    plt.xlabel('LSTAT')
    plt.ylabel('MEDV')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # fit regressor after exponential transformation
    X_log = np.log(X)
    y_sqrt = np.sqrt(y)
    regressor = LinearRegression().fit(X_log, y_sqrt)

    # compute scores
    print('[R2 score]')
    score_original = r2_score(y, (regressor.predict(X_log))**2)
    score_transform = r2_score(y_sqrt, regressor.predict(X_log))
    print('original space:{score:.3f}'.format(score=score_original))
    print('transformed space:{score:.3f}'.format(score=score_transform))

    # plot prediction of regressor
    X_pred = np.log(np.arange(X.min(), X.max(), 1))[:, np.newaxis]
    y_pred = regressor.predict(X_pred)
    plt.scatter(X.flatten(), y, label='training data', edgecolor='white')
    plt.plot(np.exp(X_pred), y_pred**2, label='prediction')
    # set plot area
    plt.xlabel('LSTAT')
    plt.ylabel('MEDV')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
