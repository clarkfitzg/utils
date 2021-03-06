'''
utils.py

General purpose python utility functions and recipes

Clark Fitzgerald
'''

import os
import csv
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


def to_csv(filename, iterable, header=None, mode='w'):
    '''
    Write an iterable to a CSV file
    '''

    with open(filename, mode) as f:
        pen = csv.writer(f)

        # Not all iterables have a next method. Example: range
        content = iter(iterable)

        # Check the length of the first row to make sure it matches
        # the length of the header
        if header:

            # Don't want to check length of string
            if isinstance(header, str):
                header = [header]

            first = next(content)

            if len(first) != len(header):
                raise ValueError('The length of the first element does '
                                 'not match the length of the header.')
            else:
                pen.writerows([header, first])

        pen.writerows(content)


def from_csv(filename, header=True):
    '''
    Returns an iterator over a CSV file
    '''
    pass


def search_replace(filename, old, new):
    '''
    Search through a file and replace string 'old' with 'new'

    Currently only supports exact matches
    '''

    # Silently skip directories
    if os.path.isdir(filename):
        return

    backup_name = filename + '.bak'
    os.rename(filename, backup_name)

    with open(backup_name) as oldfile, open(filename, 'w') as newfile:
        for line in oldfile:
            newfile.write(line.replace(old, new))

    os.remove(backup_name)


def rstrip(filename):
    '''
    Strips all white space from the right side of every line in a file.
    Useful for pep8 compliance.
    '''
    backup_name = filename + '.bak'
    os.rename(filename, backup_name)

    with open(backup_name) as oldfile, open(filename, 'w') as newfile:
        newfile.write('\n'.join(line.rstrip() for line in oldfile))

    os.remove(backup_name)


def download_many(urls):
    '''
    Download content from many urls at the same time using threading
    '''
    pass
