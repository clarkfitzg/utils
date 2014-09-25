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


class bootstrap(object):
    '''
    Implements bootstrap by calling a statistical function on
    a sample of the same size as the data with replacement.

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
        array with shape (reps, 1) holding results of bootstrapped
        statistic
    stderror : numeric
        The sample standard error of the bootstrapped statistic.
        This is the standard deviation of `results` attribute.

    Methods
    -------
    confidence(percent, method='percentile')
        Confidence interval for the statistic
    waldtest(hypothesis)
        Computes Wald test that statistic is near hypothesis
    pvalue(hypothesis)
        Computes p-value 

    References
    ----------
    Wasserman, All of Statistics, 2005
    '''

    def __init__(self, data, stat=np.mean, reps=1000, lazy=False):
        self.data = data
        self.samplesize = len(data)
        self.stat = stat
        self.reps = reps
        self._reps_remain = reps
        self.actual = stat(data)
        self.lazy = lazy
        if not lazy:
            self._run()

    def __repr__(self):
        return ''.join(['bootstrap(data, stat=', self.stat.__name__, ', reps=',
                        str(self.reps), ', lazy=', str(self.lazy), ')'])

    def __iter__(self):
        return self

    def __next__(self):
        if self._reps_remain <= 0:
            raise StopIteration
        else:
            # Decrement and return statistic applied to bootstrap sample
            self._reps_remain -= 1
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
        self.stderror = np.std(self.results)

    def _notlazy(func):
        '''
        Decorator to raise AttributeError with informative error message in
        case users try to use code which is not lazy.
        '''
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AttributeError:
                raise AttributeError("{} is not available. Try using bootstrap"
                                     " with lazy=False.".format(func.__name__))
        return wrapper

    @_notlazy
    def waldtest(self, hypothesis):
        '''
        Computes the Wald test as compared to the normal distribution:

        W = (actual - hypothesis) / stderror

        where actual is the value of the observed statistic.

        Parameters
        ----------
        hypothesis : float
            statistical parameter that's being tested against

        Returns
        -------
        W : float
            Wald test statistic

        http://en.wikipedia.org/wiki/Wald_test
        '''
        return (self.actual - hypothesis) / self.stderror

    @_notlazy
    def pvalue(self, hypothesis):
        '''
        Computes the p-value that the observed statistic is the same as
        the hypothesis.

        Parameters
        ----------
        hypothesis : float
            statistical parameter that's being tested against

        Returns
        -------
        p-value : float

        Notes
        -----
        This assumes that the statistic is normally distributed.

        A commonly used evidence scale:

        p-value         evidence
        ============================================================
        < 0.01          very strong evidence against hypothesis
        0.01 - 0.05     strong evidence against hypothesis
        0.05 - 0.1      weak evidence against hypothesis
        > 0.1           little to no evidence against hypothesis

        Source : Wasserman, 2005, All of Statistics
        '''
        W = abs(self.waldtest(hypothesis))
        return 2 * stats.norm.sf(W)

    @_notlazy
    def confidence(self, percent=95, method='percentile'):
        '''
        Compute a confidence interval.

        Parameters
        ----------
        percent : numeric
            Number between 0 and 100 indicating desired size of interval
        method : string
            'percentile' : Uses percentile function applied to empirical
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
        array([-0.10479162,  0.02798614])
        >>> b.confidence(99)
        array([-0.25964484,  0.2575893 ])

        Different methods will produce different results.

        >>> b.confidence(99, method='normal')
        array([-0.29159177,  0.2225721 ])

        '''

        allmethods = {'percentile', 'normal'}

        if method not in allmethods:
            raise NotImplementedError('{} is not an available method for'
                                      ' confidence intervals. Try one of'
                                      ' {}.'.format(method, allmethods))

        if method == 'percentile':
            diff = (100 - percent) / 2
            return np.percentile(self.results, [diff, 100 - diff])

        elif method == 'normal':
            alpha = percent / 100
            normdist = stats.norm(self.actual, self.stderror)
            return np.array(normdist.interval(alpha))


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


def rv_discrete_factory(a):
    '''
    Creates a discrete random variable from scipy.stats

    >>> rv = rv_discrete_factory((0, 0.5, 0.5, 1))
    >>> rv.cdf(0.9)
    0.75

    '''
    vals, counts = np.unique(a, return_counts=True)
    probs = counts / sum(counts)
    return stats.rv_discrete(values=(vals, probs))
