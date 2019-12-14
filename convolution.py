"""
naive implementation of convolution
"""

import numpy as np
import scipy.signal


def main():
    test_conv1d()
    test_conv2d()


def test_conv1d():
    x = [1, 3, 2, 4, 5, 6, 1, 3]
    w = [1, 0, 3, 1, 2]

    print('conv1d      : x * w ->\n', conv1d(x, w, p=2, s=1))
    print('np.convolve : x * w ->\n', np.convolve(x, w, mode='same'))


def test_conv2d():
    X = [[1, 3, 2, 4], [5, 6, 1, 3], [1, 2, 0, 2], [3, 4, 3, 2]]
    W = [[1, 0, 3], [1, 2, 1], [0, 1, 1]]

    print('conv2d                  : X * W ->\n', conv2d(X, W, p=(1, 1), s=(1, 1)))
    print('scipy.signal.convolve2d : X * W ->\n', scipy.signal.convolve2d(X, W, mode='same'))


def conv1d(x, w, p=0, s=1):
    """
    compute 1-dimension convolution

    # Parameters
    -----
    * x : array-like, shape = (n, )
        input data
    * w : array-like, shape = (m, )
        kernel of convolution
    * p : int
        padding size
    * s : int
        stride

    # Returns
    -----
    * _ : array-like, shape = (o, )
        convolution of x and w

    # Notes
    -----
    * n represents the size of input.
    * m represents the size of kernel.
    * o represents the size of output.
    """

    n, m = len(x), len(w)

    w_reversed = np.array(w[::-1])
    x_padded = np.zeros(n + 2*p)
    x_padded[p:p + n] = np.array(x)

    o = (n + 2*p - m) // s + 1
    convolution = list(np.sum(x_padded[i:i + m] * w_reversed) for i in range(0, o, s))

    return np.array(convolution)


def conv2d(X, W, p=(0, 0), s=(1, 1)):
    """
    compute 2-dimension convolution

    # Parameters
    -----
    * X : array-like, shape = (n1, n2)
        input data
    * W : array-like, shape = (m1, m2)
        kernel of convolution
    * p : tuple
        padding size for each axes
    * s : tuple
        stride for each axes

    # Returns
    -----
    * _ : array-like, shape = (o1, o2)
        convolution of X and W

    # Notes
    -----
    * (n1, n2) represents the size of input.
    * (m1, m2) represents the size of kernel.
    * (o1, o2) represents the size of output.
    """

    W_reversed = np.array(W)[::-1, ::-1]

    p1, p2 = p
    X = np.array(X)
    n1 = X.shape[0] + 2*p1
    n2 = X.shape[1] + 2*p2
    X_padded = np.zeros((n1, n2))
    X_padded[p[0]:p[0] + X.shape[0], p[1]:p[1] + X.shape[1]] = X

    m1, m2 = W_reversed.shape[0], W_reversed.shape[1]
    s1, s2 = s
    o1 = (n1 - m1) // s1 + 1
    o2 = (n2 - m2) // s2 + 1
    convolution = list(np.sum(X_padded[i1:i1 + m1, i2:i2 + m2] * W_reversed) for i1 in range(0, o1, s1) for i2 in range(0, o2, s2))

    return np.array(convolution).reshape(o1, o2)


if __name__ == '__main__':
    main()
