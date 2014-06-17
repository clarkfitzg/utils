#!/usr/bin/env python
'''
Script must be run from directory where it's located
'''

import os


# Manually appending path until bashrc is sourced
os.sys.path.append(os.sep + 'python')

import utils

wd = os.getcwd()
home = os.path.expanduser('~')
 

# TODO- Generalize this function to symlink from somewhere to anywhere

def symlink_dotfiles():
    '''
    NOTE- THIS WILL ERASE DOTFILES IN YOUR CURRENT HOME DIRECTORY!!!

    Make symbolic links for each file and put them in the home directory
    '''

    dotfiles = wd + os.sep + 'dotfiles'
   
    for file in os.listdir(dotfiles):
        src = dotfiles + os.sep + file
        dst = home + os.sep + '.' + file
        if os.path.isfile(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(src, dst)


if __name__ == '__main__':
    symlink_dotfiles()
