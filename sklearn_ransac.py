"""
show sample of how to use RANSACRegressor
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

from data_utility import HousingData
from plot_utility import plot_predictions


def main():
    # prepare training data and target variable
    features = ['RM']
    D = HousingData(features)
    X, y = D.X, D.y

    # prepare and fit RANSAC regressor
    ransac = RANSACRegressor(
        LinearRegression(),
        max_trials=100,
        min_samples=50,
        loss='absolute_loss',
        residual_threshold=5.0,
        random_state=0)
    ransac.fit(X, y)

    # show prediction
    plot_predictions(
        X.flatten(), y, ransac,
        xlabel='RM', ylabel='MEDV')

    # plot inliers and outliers
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_X = np.arange(3, 10, 1)
    line_y = ransac.predict(line_X[:, np.newaxis])
    plt.scatter(
        X[inlier_mask], y[inlier_mask],
        c='steelblue', edgecolor='white', marker='o', label='inliers')
    plt.scatter(
        X[outlier_mask], y[outlier_mask],
        c='limegreen', edgecolor='white', marker='s', label='outliers')
    plt.plot(line_X, line_y, color='black')
    plt.xlabel('RM')
    plt.ylabel('MEDV')
    plt.legend()
    plt.show()

    # show a sample of non-standardized prediction and weights
    rm = [[5.0]]
    medv = ransac.predict(rm)[0]
    print('RM = 5.0 -> MEDV = {:.3e}'.format(medv))

    # show weights
    print('intercept = {i:.3e}, slope = {s:.3e}'.format(i=ransac.estimator_.intercept_, s=ransac.estimator_.coef_[0]))


if __name__ == '__main__':
    main()
