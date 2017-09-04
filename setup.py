#!/usr/bin/env python
'''
Script must be run from within directory where it's located
so that working directory is correct.
'''

import os


wd = os.getcwd()
home = os.path.expanduser('~')


def symlink_dotfiles():
    '''
    NOTE- THIS WILL ERASE DOTFILES IN YOUR CURRENT HOME DIRECTORY!!!

    Make symbolic links for each file and put them in the home directory
    '''

    dotfiles = wd + os.sep + 'dotfiles'

    for f in os.listdir(dotfiles):
        src = dotfiles + os.sep + f
        dst = home + os.sep + '.' + f
        if os.path.isfile(dst) or os.path.islink(dst):
            #print('removing {}'.format(dst))
            os.remove(dst)
        os.symlink(src, dst)
        #print('linking to new {}'.format(dst))


if __name__ == '__main__':
    symlink_dotfiles()
