"""
show sample of how to use cnn_layers_v1
"""

import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cnn_layers_v1 import CNN

IMAGE_SHAPE = (8, 8)


def main():
    # prepare sample data and target variable
    X, y = load_digits(return_X_y=True)

    # split sample data into training data and test data and standardize them
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=1, stratify=y_train)
    sc = StandardScaler().fit(X_train)
    X_train_std = sc.transform(X_train)
    X_validation_std = sc.transform(X_validation)
    X_test_std = sc.transform(X_test)

    # set parameters
    input_height, input_width = IMAGE_SHAPE
    n_outputs = len(np.unique(y))
    learning_rate = 1.0e-4
    random_seed = 123

    # construct CNN
    cnn = CNN(
        input_height, input_width, n_outputs,
        learning_rate=learning_rate, random_seed=random_seed)

    # train CNN
    cnn.train(
            training_set=(X_train_std, y_train),
            validation_set=(X_validation_std, y_validation),
            initialize=True)

    # predict test data
    y_pred = cnn.predict(X_test_std)
    print('test accuracy: {0:.3f}'.format(np.sum(y_pred == y_test) / len(y_test)))

    # further train CNN
    cnn.train(
            training_set=(X_train_std, y_train),
            validation_set=(X_validation_std, y_validation),
            initialize=False)

    # predict test data
    y_pred = cnn.predict(X_test_std)
    print('test accuracy: {0:.3f}'.format(np.sum(y_pred == y_test) / len(y_test)))


if __name__ == '__main__':
    main()
