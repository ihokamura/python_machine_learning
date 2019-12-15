"""
show sample of how to use cnn_nn_v1
"""

import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cnn_nn_v1 import construct_cnn, construct_convolution_layer, construct_fc_layer, load, predict, save, train

IMAGE_SHAPE = (8, 8)


def test_layer():
    # test construct_convolution_layer()
    print('[test construct_convolution_layer()]')
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32, shape=[None, 8, 8, 1])
        construct_convolution_layer(x, name='test_construct_convolution_layer', kernel_shape=(3, 3), n_output_channels=32)
    del g, x

    # test construct_fc_layer()
    print('[test construct_fc_layer()]')
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32, shape=[None, 8, 8, 1])
        construct_fc_layer(x, name='test_construct_fc_layer', n_output_units=32, activation_function=tf.nn.relu)
    del g, x


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

    # construct computation graph
    g = tf.Graph()
    with g.as_default():
        # construct CNN
        tf.set_random_seed(random_seed)
        construct_cnn(input_height, input_width, n_outputs, learning_rate)
        # define saver
        saver = tf.train.Saver()
        # save computation graph for tensorboard
        tf.summary.FileWriter(logdir='./logs/', graph=g)

    # train and save model
    with tf.Session(graph=g) as sess:
        train(
            sess,
            training_set=(X_train_std, y_train),
            validation_set=(X_validation_std, y_validation),
            initialize=True,
            random_seed=random_seed)
        save(saver, sess, epoch=20)

    # reconstruct computation graph to demonstrate how to use load()
    del g
    g = tf.Graph()
    with g.as_default():
        # construct CNN
        tf.set_random_seed(random_seed)
        construct_cnn(input_height, input_width, n_outputs, learning_rate)
        # define saver
        saver = tf.train.Saver()

    # load model and predict test data
    with tf.Session(graph=g) as sess:
        load(saver, sess, epoch=20)
        y_pred = predict(sess, X_test_std)
        print('test accuracy: {0:.3f}'.format(np.sum(y_pred == y_test) / len(y_test)))

    # further train model and predict test data
    with tf.Session(graph=g) as sess:
        load(saver, sess, epoch=20)
        train(
            sess,
            training_set=(X_train_std, y_train),
            validation_set=(X_validation_std, y_validation),
            initialize=False,
            random_seed=random_seed)
        save(saver, sess, epoch=40)
        y_pred = predict(sess, X_test_std)
        print('test accuracy: {0:.3f}'.format(np.sum(y_pred == y_test) / len(y_test)))


if __name__ == '__main__':
    test_layer()
    main()
