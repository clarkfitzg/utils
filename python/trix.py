'''
trix.py

Utility functions for working with matrices in Numpy

Clark Fitzgerald
'''

from __future__ import division
import numpy as np


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
