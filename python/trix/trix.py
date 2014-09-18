'''
trix.py

Utility functions for working with matrices in Numpy
Extending to other statistics

Author: Clark Fitzgerald
License: BSD 3-clause
'''

from __future__ import division

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


class bootstrap(object):
    '''
    Implements a statistical bootstrap by calling statistic on a sample
    of the same size as the data with replacement.

    Parameters
    ----------
    data : ndarray
        sample data
    stat : callable
        function to call on data
    reps : int
        Number of repetitions
    lazy : boolean
        If lazy = True then this object is an iterator, returning the
        statistic called on a new sample each time.

    Attributes
    ----------
    actual : numeric
        The actual value of the statistic called on the data
    results : ndarray
        sorted array with shape (reps, 1) holding results of bootstrapped
        statistic
        

    References
    ----------
    Wasserman, All of Statistics, 2005
    '''

    def __init__(self, data, stat=np.mean, reps=10, lazy=False):
        self.data = data
        self.samplesize = len(data)
        self.stat = stat
        self.reps = reps
        self.reps_remain = reps
        self.actual = stat(data)
        if not lazy:
            self._run()

    def __iter__(self):
        return self

    def __next__(self):
        if self.reps_remain <= 0:
            raise StopIteration
        else:
            # Decrement and return statistic applied to bootstrap sample
            self.reps_remain -= 1
            bootsample = np.random.choice(self.data, self.samplesize)
            return self.stat(bootsample)

    def __len__(self):
        return self.reps

    def _run(self):
        '''
        Run the bootstrap simulation.

        If lazy = False (the default) then this will run on instantiation,
        creating the results arribute.
        '''
        self.results = np.array([stat for stat in self])
        self.results.sort()

    def stderr(self):
        '''
        Compute the sample standard error of the bootstrapped statistic.
        This is the standard deviation of `results` attribute.

        Returns
        -------
        std_error : float

        Examples
        --------
        >>> np.random.seed(321)
        >>> b = bootstrap(np.random.randn(100), stat=np.mean, reps=100)
        >>> b.stderr()
        0.099805501974072466

        '''
        try:
            return np.std(self.results)
        except AttributeError:
            raise AttributeError("The bootstrap results are not available. "
                                 "Try using bootstrap with lazy=False.")

    def waldtest(self, hypothesis):
        '''

        Parameters
        ----------
        hypothesis : float
            statistical parameter that's being tested against
        '''
        pass

    def confidence(self, percent=95, method='percentile'):
        '''
        Compute a confidence interval. 

        Parameters
        ----------
        percent : numeric
            Number between 0 and 100 indicating desired size of interval
        method : string
            'percentile' : Uses percentage point function applied to empirical
                distribution of results.
            'normal' : assumes that the distribution of the bootstrapped 
                statistic is normal.
            'pivotal' : Not yet implemented

        Returns
        -------
        lower, upper : ndarray of float
            lower and upper bounds for the confidence interval

        See Also
        --------
        stats.<distribution>.interval : Compute exact interval around median
        when distribution is known.

        Examples
        --------
        >>> np.random.seed(321)
        >>> b = bootstrap(np.random.randn(100), stat=np.mean, reps=100)

        Higher confidence generally implies larger intervals.

        >>> b.confidence(50)
        array([-0.70899958,  0.63997992])
        >>> b.confidence(99)
        array([-2.61033914,  2.54131947])

        Different methods will produce different results.

        >>> b.confidence(99, method='normal')
        array([-2.61033914,  2.54131947])

        '''
        alpha = percent / 100

        if method == 'percentile':
            pass

        elif method == 'normal':
            return self.actual + np.array(stats.norm.interval(alpha))


def frac_above(array, bound, how='strict'):
    '''
    What fraction of the array is above the bound?

    >>> frac_above(np.array([0, 1, 3, 5]), 1)
    0.5

    '''
    if how == 'strict':
        return sum(array > bound) / array.size


if __name__ == '__main__':

    b = bootstrap(np.random.randn(100))
