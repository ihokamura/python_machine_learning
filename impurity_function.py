"""
show graph of impurity functions
"""

import matplotlib.pyplot as plt
import numpy as np


# Gini impurity
def gini(p):
    return 1 - (p**2 + (1 - p)**2)


# entropy
def entropy(p):
    if (p == 0) or (p == 1):
        return 0
    else:
        return -(p*np.log2(p) + (1 - p)*np.log2(1 - p))


# classification error
def error(p):
    return 1 - max(p, 1 - p)


def main():
    x_points = np.arange(0, 1, 0.01)
    y_gini = [gini(x) for x in x_points]
    y_entropy = [0.5*entropy(x) for x in x_points]
    y_error = [error(x) for x in x_points]

    ax = plt.subplot(111)
    ax.axhline(y=0.5, linewidth=1, linestyle='--', color='k')
    for y_points, label in zip(
        [y_gini, y_entropy, y_error],
        ['Gini impurity', 'entropy', 'classification error']
        ):
        ax.plot(x_points, y_points, label=label)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True, shadow=False)
    plt.xlabel('p')
    plt.ylabel('impurity index')
    plt.show()


if __name__ == '__main__':
    main()
