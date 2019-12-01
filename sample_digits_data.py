"""
show sample of how to use load_digits
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def show_digits_data():
    # load digits dataset
    digits = load_digits()

    # show some members
    print('shape of data:', digits.data.shape)
    print('shape of images:', digits.images.shape)
    print('shape of target:', digits.target.shape)

    # show some images
    nrows, ncols = 5, 5
    _, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    ax = ax.flatten()
    for axis, image, label in zip(ax, digits.images, digits.target):
        axis.imshow(image, cmap='Greys')
        axis.set_title(label)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    show_digits_data()
