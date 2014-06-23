'''
utils.py

General purpose python utility functions

Clark Fitzgerald
'''

import functools


def replicate(n, func, *args, **kwargs):
    '''
    Returns a generator calling func n times.
    Emulates `replicate` from R.
    Useful for simulating random events.

    >>> list(replicate(3, pow, 2, 2))
    [4, 4, 4]

    See also:
        itertools.islice
    '''
    return (func(*args, **kwargs) for i in range(n))
