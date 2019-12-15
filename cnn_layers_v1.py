"""
implement CNN with tf.compat.v1.layers
"""

import os

import numpy as np
import tensorflow.compat.v1 as tf

from cnn_utility import generate_batch


class CNN:
    """
    CNN (convolution neural network) classifier for 2-D image

    # Parameters
    -----
    * input_height : int
        number of pixels in vertical direction
    * input_width : int
        number of pixels in horizontal direction
    * n_outputs : int
        number of outputs (class labels)
    * batch_size : int
        size of minibatch
    * epochs : int
        number of epochs
    * learning_rate : float
        learning rate of Adam optimizer
    * dropout_rate : float
        dropout rate
    * shuffle : bool
        indicator to check if it is necessary to shuffle data
    * random_seed : int or `None`
        random generator seed

    # Attributes
    -----
    training_loss_ : list
        values of cost function at each epoch
    """

    def __init__(
        self,
        input_height, input_width, n_outputs, 
        batch_size=64, epochs=20,
        learning_rate=1.0e-4, dropout_rate=0.5, shuffle=True, random_seed=None):
        # initialize parameters
        self.input_height = input_height
        self.input_width = input_width
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle
        self.random_seed = random_seed

        # construct computation graph
        g = tf.Graph()
        with g.as_default():
            tf.set_random_seed(self.random_seed)
            self.build()
            self._init_operator = tf.global_variables_initializer()

        # make a session
        self._sess = tf.Session(graph=g)

    def build(self):
        """
        construct CNN

        # Parameters
        -----
        * (no parameters)

        # Returns
        -----
        * None
        """

        # define placeholder for X and y
        tf_X = tf.placeholder(tf.float32, shape=[None, self.input_height * self.input_width], name='tf_X')
        tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')
        is_train = tf.placeholder(tf.bool, shape=[], name='is_train')

        # convert X into 4-D tensor
        tf_X_image = tf.reshape(tf_X, shape=[-1, self.input_height, self.input_width, 1], name='tf_X_image')

        # encode y by one-hot representation
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=self.n_outputs, dtype=tf.float32, name='tf_y_onehot')

        # construct CNN
        # 1st layer: convolution layer 1 and maximum pooling layer
        h1 = tf.layers.conv2d(inputs=tf_X_image, filters=32, kernel_size=(3, 3), padding='valid', activation=tf.nn.relu)
        h1_pool = tf.layers.max_pooling2d(h1, pool_size=(2, 2), strides=(2, 2), padding='same')

        # 2nd layer: convolution layer 2 and maximum pooling layer
        h2 = tf.layers.conv2d(inputs=h1_pool, filters=64, kernel_size=(3, 3), padding='valid')
        h2_pool = tf.layers.max_pooling2d(h2, pool_size=(2, 2), strides=(2, 2), padding='same')

        # 3rd layer: fully connected layer 1
        n_input_units = np.prod(h2_pool.get_shape().as_list()[1:])
        h2_pool_flat = tf.reshape(h2_pool, shape=[-1, n_input_units])
        h3 = tf.layers.dense(inputs=h2_pool_flat, units=1024, activation=tf.nn.relu)

        # dropout
        h3_dropout = tf.layers.dropout(inputs=h3, rate=self.dropout_rate, training=is_train)

        # 4th layer: fully connected layer 2
        h4 = tf.layers.dense(inputs=h3_dropout, units=self.n_outputs, activation=None)

        # predict class label
        predictions = {
            'probabilities':tf.nn.softmax(h4, name='probabilities'),
            'labels':tf.cast(tf.argmax(h4, axis=1), tf.int32, name='labels')}

        # define cost function and optimizer
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4, labels=tf_y_onehot), name='cross_entropy_loss')
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss, name='train_optimizer')

        # compute accuracy
        correct_predictions = tf.equal(predictions['labels'], tf_y, name='correct_predictions')
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    def train(self, training_set, validation_set=None, initialize=True):
        """
        train CNN model

        # Parameters
        -----
        * training_set : tuple
            tuple of training dataset of input and output variables
        * validation_set : tuple or `None`
            tuple of validation dataset of input and output variables
        * initialize : bool
            indicator to check if it is necessary to initialize parameters(weights)

        # Returns
        -----
        * None
        """

        X_data = np.array(training_set[0])
        y_data = np.array(training_set[1])
        training_loss = []
        sess = self._sess

        # initialize variables
        if initialize:
            sess.run(self._init_operator)

        # train model for given epochs
        for epoch in range(1, self.epochs + 1):
            batch = generate_batch(X_data, y_data, shuffle=self.shuffle)
            sum_of_loss = 0.0

            for i, (batch_X, batch_y) in enumerate(batch):
                feed = {'tf_X:0':batch_X, 'tf_y:0':batch_y, 'is_train:0':True}
                loss, _ = sess.run(['cross_entropy_loss:0', 'train_optimizer'], feed_dict=feed)
                sum_of_loss += loss

            # save and show the mean of loss over a minibatch
            training_loss.append(sum_of_loss/(i + 1))
            print('epoch {0:2d} training average loss: {1:7.3f}'.format(epoch, sum_of_loss), end=' ')

            if validation_set is not None:
                # show accuracy for validation data
                feed = {'tf_X:0':validation_set[0], 'tf_y:0':validation_set[1], 'is_train:0':False}
                validation_accuracy = sess.run('accuracy:0', feed_dict=feed)
                print(' validation accuracy: {0:7.3f}'.format(validation_accuracy))
            else:
                print(' ')

        # save loss
        self.training_loss_ = training_loss

    def predict(self, X, return_proba=False):
        """
        predict by CNN model

        # Parameters
        -----
        * X : array-like, shape = (n_samples, n_features)
            test data
        * return_proba : bool
            indicator to return probabilities

        # Returns
        -----
        * _ : float or array-like, shape = (n_outputs)
            If return_proba is `True`, the function returns probabilities in which class labels training data are classified.
            Otherwise, the function returns class labels of test data.

        # Notes
        -----
        * n_samples represents the number of samples.
        * n_features represents the number of features.
        * n_outputs represents the number of outputs (class labels).
        """

        feed = {'tf_X:0':X, 'is_train:0':False}
        if return_proba:
            return self._sess.run('probabilities:0', feed_dict=feed)
        else:
            return self._sess.run('labels:0', feed_dict=feed)
