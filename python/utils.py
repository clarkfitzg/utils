'''
utils.py

General purpose python utility functions

Clark Fitzgerald
'''


from __future__ import division
import itertools
import functools


def replicate(n, func, *args, **kwargs):
    '''
    Returns a generator calling func n times.
    Basically copying `replicate` from the R language.

    >>> list(replicate(3, pow, 2, 2))
    [4, 4, 4]

    '''
    pfunc = functools.partial(func, *args, **kwargs)
    for i in range(n):
        yield pfunc()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
