#!/usr/bin/env python

'''
rstrip.py

Use to make Python files Pep8 compliant by stripping white space on right
'''

import os
import sys


def filerstrip(filename):
    '''
    Runs rstrip on each line in a file
    '''

    print('Stripping white space from {}'.format(filename))

    backup = filename + '.bak'
    os.rename(filename, backup)

    with open(backup) as old, open(filename, 'w') as new:
        for line in old:
            print(line.rstrip(), file=new)

    os.remove(backup)


if __name__ == '__main__':
    filenames = sys.argv[1:]

    for f in filenames:
        filerstrip(f)
