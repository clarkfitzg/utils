'''
utils.py

General purpose python utility functions

Clark Fitzgerald
'''

import functools
import collections


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

    >>> a = weighcount({'a': 2, 'b': 8})
    >>> a.total()
    10
    >>> dict(a.weights())
    {'b': 0.8, 'a': 0.2}

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
        pass

    def weights(self, dynamic=False):
        '''
        Generator over (element, weight) tuples
        '''
        for key in self:
            yield key, self[key] / self.total()

    def common_weights(self, n=None):
        '''
        List the n most common elements and their weights from the most
        common to the least.  If n is None, then list all element counts.

        >>> weighcount('abcdeabcdabcaba').common_weights(3)

        Analagous to `most_common` method for counts.
        '''
        pass


    







