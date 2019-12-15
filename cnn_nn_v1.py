"""
implement CNN with tf.compat.v1.nn
"""

import os

import numpy as np
import tensorflow.compat.v1 as tf

from cnn_utility import generate_batch


def construct_convolution_layer(input_tensor, name, kernel_shape, n_output_channels, padding_mode='SAME', strides=(1, 1, 1, 1)):
    """
    construct convolution layer of CNN

    # Parameters
    -----
    * input_tensor : Tensor, shape = (batch_size, input_height, input_width, n_input_channels)
        input tensor of the convolution layer
    * name : str
        name of variable scope
    * kernel_shape : iterable
        shape of kernel (weight)
    * n_output_channels : int
        number of output channels
    * padding_mode : str
        type of padding algorithm
        One of the following must be used.
        * 'SAME'
        * 'VALID'
    * strides : tuple
        strides for each axes

    # Returns
    -----
    * convolution : Tensor, shape = (batch_size, output_height, output_width, n_input_channels)
        output of the convolution layer

    # Notes
    -----
    * n_input_channels represents the number of input channels.
    """

    with tf.variable_scope(name):
        # define weights
        n_input_channels = input_tensor.shape[-1]
        weights_shape = list(kernel_shape) + [n_input_channels, n_output_channels]
        weights = tf.get_variable(name='_weights', shape=weights_shape)
        print(weights)

        # define biases
        biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[n_output_channels]))
        print(biases)

        # compute convolution
        convolution = tf.nn.conv2d(input=input_tensor, filter=weights, strides=strides, padding=padding_mode)
        print(convolution)
        convolution = tf.nn.bias_add(convolution, biases, name='pre_activation')
        print(convolution)
        convolution = tf.nn.relu(convolution, name='activation')
        print(convolution)

        return convolution


def construct_fc_layer(input_tensor, name, n_output_units, activation_function=None):
    """
    construct fully connected layer of CNN

    # Parameters
    -----
    * input_tensor : Tensor, shape = (batch_size, input_height, input_width, n_input_channels)
        input tensor of the fully connected layer
    * name : str
        name of variable scope
    * n_output_units : int
        number of output units
    * activation_function : function or `None`
        activation function
        If it is None, no activation is applied.

    # Returns
    -----
    * convolution : Tensor, shape = (batch_size, output_height, output_width, n_input_channels)
        output of the convolution layer

    # Notes
    -----
    * n_input_channels represents the number of input channels.
    """

    with tf.variable_scope(name):
        # flatten input tensor
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))

        # define weights
        weights_shape = [n_input_units, n_output_units]
        weights = tf.get_variable(name='_weights', shape=weights_shape)
        print(weights)

        # define biases
        biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[n_output_units]))
        print(biases)

        # compute FC layer
        layer = tf.matmul(input_tensor, weights)
        print(layer)
        layer = tf.nn.bias_add(layer, biases, name='pre_activation')
        print(layer)
        if activation_function is None:
            return layer
        else:
            layer = activation_function(layer, name='activation')
            print(layer)
            return layer


def construct_cnn(input_height, input_width, n_outputs, learning_rate=0.001):
    """
    construct CNN

    # Parameters
    -----
    * input_height : int
        number of pixels in vertical direction
    * input_width : int
        number of pixels in horizontal direction
    * n_outputs : int
        number of outputs (class labels)
    * learning_rate : float
        learning rate of Adam optimizer

    # Returns
    -----
    * None
    """

    # define placeholder for X and y
    tf_X = tf.placeholder(tf.float32, shape=[None, input_height * input_width], name='tf_X')
    tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')

    # convert X into 4-D tensor
    tf_X_image = tf.reshape(tf_X, shape=[-1, input_height, input_width, 1], name='tf_X_image')

    # encode y by one-hot representation
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=n_outputs, dtype=tf.float32, name='tf_y_onehot')

    # construct CNN
    # 1st layer: convolution layer 1 and maximum pooling layer
    print('building 1st layer...')
    h1 = construct_convolution_layer(tf_X_image, name='conv1', kernel_shape=(3, 3), padding_mode='VALID', n_output_channels=32)
    h1_pool = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 2nd layer: convolution layer 2 and maximum pooling layer
    print('building 2nd layer...')
    h2 = construct_convolution_layer(h1_pool, name='conv2', kernel_shape=(3, 3), padding_mode='VALID', n_output_channels=64)
    h2_pool = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 3rd layer: fully connected layer 1
    print('building 3rd layer...')
    h3 = construct_fc_layer(h2_pool, name='fc3', n_output_units=1024, activation_function=tf.nn.relu)

    # dropout
    keep_probability = tf.placeholder(tf.float32, name='fc_keep_probability')
    h3_dropout = tf.nn.dropout(h3, keep_prob=keep_probability, name='dropout_layer')

    # 4th layer: fully connected layer 2
    print('building 4th layer...')
    h4 = construct_fc_layer(h3_dropout, name='fc4', n_output_units=n_outputs, activation_function=None)

    # predict class label
    predictions = {
        'probabilities':tf.nn.softmax(h4, name='probabilities'),
        'labels':tf.cast(tf.argmax(h4, axis=1), tf.int32, name='labels')}

    # define cost function and optimizer
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4, labels=tf_y_onehot), name='cross_entropy_loss')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name='train_optimizer')

    # compute accuracy
    correct_predictions = tf.equal(predictions['labels'], tf_y, name='correct_predictions')
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')


def train(sess, training_set, validation_set=None, initialize=True, epochs=20, shuffle=True, dropout=0.5, random_seed=None):
    """
    train CNN model

    # Parameters
    -----
    * sess : tf.Session
        session for running operation
    * training_set : tuple
        tuple of training dataset of input and output variables
    * validation_set : tuple or `None`
        tuple of validation dataset of input and output variables
    * initialize : bool
        indicator to check if it is necessary to initialize parameters(weights)
    * epochs : int
        number of epochs at which the model is saved
    * shuffle : bool
        indicator to check if it is necessary to shuffle batch
    * dropout : float
        probability of dropout
    * random_seed : int or `None`
        random generator seed

    # Returns
    -----
    * None
    """

    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss = []

    # initialize variables
    if initialize:
        sess.run(tf.global_variables_initializer())

    # initialize random generator
    np.random.seed(random_seed)

    # train model for given epochs
    for epoch in range(1, epochs + 1):
        batch = generate_batch(X_data, y_data, shuffle=shuffle)
        sum_of_loss = 0.0

        for i, (batch_X, batch_y) in enumerate(batch):
            feed = {'tf_X:0':batch_X, 'tf_y:0':batch_y, 'fc_keep_probability:0':dropout}
            loss, _ = sess.run(['cross_entropy_loss:0', 'train_optimizer'], feed_dict=feed)
            sum_of_loss += loss

        # save and show the mean of loss over a minibatch
        training_loss.append(sum_of_loss/(i + 1))
        print('epoch {0:2d} training average loss: {1:7.3f}'.format(epoch, sum_of_loss), end=' ')

        if validation_set is not None:
            # show accuracy for validation data
            feed = {'tf_X:0':validation_set[0], 'tf_y:0':validation_set[1], 'fc_keep_probability:0':1.0}
            validation_accuracy = sess.run('accuracy:0', feed_dict=feed)
            print(' validation accuracy: {0:7.3f}'.format(validation_accuracy))
        else:
            print(' ')


def predict(sess, X, return_proba=False):
    """
    predict by CNN model

    # Parameters
    -----
    * sess : tf.Session
        session for running operation
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

    feed = {'tf_X:0':X, 'fc_keep_probability:0':1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)


def save(saver, sess, epoch, path='./model/'):
    """
    save CNN model

    # Parameters
    -----
    * saver : tf.train.Saver
        model saver
    * sess : tf.Session
        session for running operation
    * epoch : int
        epoch at which the model is saved
    * path : str
        path to directory to save model

    # Returns
    -----
    * None
    """

    if not os.path.isdir(path):
        os.makedirs(path)

    print('saving model in {path} ...'.format(path=path))
    saver.save(sess, os.path.join(path, 'cnn_model.ckpt'), global_step=epoch)


def load(saver, sess, epoch, path='./model/'):
    """
    load CNN model

    # Parameters
    -----
    * saver : tf.train.Saver
        model saver
    * sess : tf.Session
        session for running operation
    * epoch : int
        epoch at which the model is loaded
    * path : str
        path to directory to load model

    # Returns
    -----
    * None
    """

    print('loading model from {path} ...'.format(path=path))
    saver.restore(sess, os.path.join(path, 'cnn_model.ckpt-{epoch}'.format(epoch=epoch)))
