'''
utils.py

General purpose python utility functions

Clark Fitzgerald
'''

import collections
from collections import Iterable


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


class weighcount(collections.Counter):
    '''
    Returns a subclass of collections.Counter that computes
    weights and totals

    >>> wc = weighcount({'a': 2, 'b': 8})
    >>> wc.total()
    10
    >>> wc.common_weights()
    [('b', 0.8), ('a', 0.2)]

    See also:
        collections.Counter
    '''

    def total(self):
        '''
        The sum of all counts
        '''
        return sum(self.values())

    def weight(self, key):
        '''
        Returns the weight associated with `key`.
        '''
        return self[key] / self.total()

    def gen_weights(self):
        '''
        Generator over (element, weight) tuples.
        Weight is recomputed for every element / iteration.
        '''
        for key in self:
            yield key, self.weight(key)

    def common_weights(self, n=None):
        '''
        List the n most common elements and their weights from the most
        common to the least.  If n is None, then list all element counts.

        >>> letters = weighcount({'a': 2, 'b': 8, 'c': 10})
        >>> letters.common_weights(2)
        [('c', 0.5), ('b', 0.4)]

        Analagous to `most_common` method for counts.
        '''
        common_values = (t[0] for t in self.most_common(n))
        return [(x, self.weight(x)) for x in common_values]


def flatten(nested_items, ignore_types=(str, bytes)):
    '''
    Generator returning elements contained in nested sequence

    Recipe 4.14 from Python Cookbook
    '''
    for x in nested_items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x, ignore_types)
        else:
            yield x
