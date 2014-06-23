# !usr/bin/python
# rstrip.py

'''
Strip all spaces from the right side and bottom of a file

Use to make Python files Pep8 compliant
'''

import os


def filerstrip(filename):

    backup = filename + '.bak'
    os.rename(filename, backup)

    with open(backup) as old:
        with open(filename, 'w') as new:
            for line in old:
                new.write('\n' + line.rstrip())
