'''
trix.py

Utility functions for working with matrices in Numpy
Extending to other statistics

Clark Fitzgerald
'''

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt


def matrixij(shape, ij):
    '''
    Returns an array with 1 in position i, j, and zeros elsewhere

    >>> matrixij((2, 3), (0, 1))
    array([[0, 1, 0],
           [0, 0, 0]])

    '''
    m = np.zeros(shape, dtype=int)
    m[ij] = 1
    return m


def matrixbasis(n, m):
    '''
    Generates a basis for a vector space of n x m matrices

    >>> list(matrixbasis(1, 2))
    [array([[1, 0]]), array([[0, 1]])]

    '''
    for i in range(n):
        for j in range(m):
            yield matrixij((n, m), (i, j))


def check_orthonormal(A):
    '''
    Return true if a 2 dimensional array A is numerically orthonormal

    >>> check_orthonormal(np.array([[0, 1], [1, 0]]))
    True

    >>> check_orthonormal(np.array([[0, 3], [1, 0]]))
    False

    '''
    if A.ndim != 2:
        raise ValueError('This function can only be called on a '
                         '2 dimensional array.')
    else:
        return np.allclose(A.dot(A.T), np.eye(A.shape[0]))


def plot_rv_cont(rv, nsamp=100, nruns=5):
    '''
    Plot probability distribution for a continous random variable.
    The line is the probability density function, the histograms are
    realizations with `nsamp` samples.

    Parameters
    ----------
    rv : frozen continuous random variable from scipy.stats
    nsamp  : number of samples for each run
    nruns  : number of times to draw nsamp and plot histogram

    '''
    plot_params = {'normed': True, 'histtype': 'stepfilled',
                   'alpha': 1.0 / nruns, 'color': 'green'}

    left = rv.median()
    right = rv.median()

    for i in range(nruns):
        samps = rv.rvs(nsamp)
        left = min(left, min(samps))
        right = max(right, max(samps))
        plt.hist(samps, **plot_params)

    # Plot pdf only where samples were realized
    x = np.linspace(left, right, num=100)
    y = rv.pdf(x)
    plt.plot(x, y, linewidth=4)

    plt.title('{} distribution'.format(rv.dist.name))
    plt.show()

    return plt
