'''
trix.py

Utility functions for working with matrices in Numpy
Extending to other statistics

Author: Clark Fitzgerald
License: BSD 3-clause
'''

from __future__ import division
import functools

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


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
    Generator yielding a basis for a vector space of n x m 2d arrays.
    Set n=1 to yield a basis of m dimensional vectors

    Parameters
    ----------
    n : int
        Number of rows in matrix
    m : int
        Number of columns in matrix

    Returns
    -------
    basis : Generator
        Yields matrix with 1 in position (i, j), and zeros elsewhere.
        Cycles through rows first.

    See Also
    --------
    matrixij : Returns single a single matrix

    Examples
    --------
    >>> list(matrixbasis(1, 2))
    [array([[1, 0]]), array([[0, 1]])]

    '''
    for i in range(n):
        for j in range(m):
            yield matrixij((n, m), (i, j))


def check_orthonormal(A):
    '''
    Returns True if a 2 dimensional array is numerically orthonormal.

    Parameters
    ----------
    A : ndarray
        array to be checked

    Returns
    -------
    orthonormal : Boolean
        True if orthonormal, otherwise False

    Examples
    --------
    >>> check_orthonormal(np.array([[0, 1], [1, 0]]))
    True
    >>> check_orthonormal(np.array([[0, 3], [1, 0]]))
    False

    '''
    if A.ndim != 2:
        raise ValueError('This function can only be called on a '
                         '2 dimensional array.')

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


def replicate(func, n, *args, **kwargs):
    '''
    Call func(*args, **kwargs) n times and return results as ndarray.
    Simulates random events, similar to R's replicate.

    Parameters
    ----------
    func : callable
        Function involving random events
    n : int
        Number of times to call `func`
    *args, **kwargs : arguments
        Additional arguments for `func`

    Returns
    -------
    results : ndarray
        Each row is the result of calling func

    Examples
    --------
    >>> from numpy.random import seed, choice
    >>> f = lambda x: sum(choice(10, size=x))
    >>> seed(23)
    >>> replicate(f, 3, 10)
    array([65, 39, 40])

    '''
    results = [func(*args, **kwargs) for i in range(n)]
    return np.array(results)


def cdf(array, x):
    '''
    Cumulative distribution function evaluated at x using array as an
    empirical probability distribution. 
    
    Imitates API of scipy.stats.<distribution>.
    '''
    pass


def empirical(array, bins=1000):
    '''
    Creates an approximate empirical probability distribution through binning.
 
    Parameters
    ----------
    array : array_like
        1 dimensional array
    bins : int
        Number of bins

    Returns
    -------
    rv : rv_discrete
        instance of scipy.stats.rv_discrete approximating the distribution
        of array

    Examples
    --------
    '''
    pass


if __name__ == '__main__':

    np.random.seed(10)
    b = bootstrap(np.random.randn(100))
