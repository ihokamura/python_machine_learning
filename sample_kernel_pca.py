"""
show sample of how to use kernel PCA transformer
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.decomposition import KernelPCA

from kernel_pca import RBFKernelPCA
from plot_utility import plot_features


def main():
    # moon-shaped dataset
    X, y = make_moons(n_samples=100, random_state=123)
    show_effect_of_kpca(X, y)

    # circular dataset
    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
    show_effect_of_kpca(X, y)


def show_effect_of_kpca(X, y):
    # plot the original dataset
    plot_features(X, y, title='original dataset', loc='upper right')

    # execute kernel PCA transform
    gammas = [1, 5, 15, 50, 100]
    for gamma in gammas:
        kpcas = [
            RBFKernelPCA(n_components=2, gamma=gamma),
            KernelPCA(n_components=2, kernel='rbf', gamma=gamma)]
        for kpca in kpcas:
            X_kpca = kpca.fit_transform(X)
            plot_features(X_kpca, y, title=r'$\gamma$ = {0}'.format(gamma), loc='upper right')


if __name__ == '__main__':
    main()
