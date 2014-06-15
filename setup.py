#!/usr/bin/env python
'''
NOTE- THIS WILL ERASE DOTFILES IN YOUR CURRENT HOME DIRECTORY!!!

Make symbolic links for each file and put them in the home directory

Script must be run from this directory
'''

import os


wd = os.getcwd()
home = os.path.expanduser('~')

# These files in the working directory will be skipped
skip = {'.git', 'add_symlinks.py', 'README.mdown'}

# Remove existing dotfiles and add symbolic links
for file in os.listdir(wd):
    if file in skip:
        continue
    src = wd + os.sep + file
    dst = home + os.sep + '.' + file
    if os.path.isfile(dst):
        os.remove(dst)
    os.symlink(src, dst)
