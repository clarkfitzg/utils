# test_rstrip.py

import os
import rstrip


class test_rstrip:

    def setup(self):
        with open('temp.txt', 'w') as f: 
            f.write('  1    \n   2  \n\n\n  third line    \n\n\n')

    def teardown(self):
        os.remove('temp.txt')

    def test_pass(self):
        pass

    def test_default_blankstrip(self):
        rstrip.blankstrip('temp.txt')
        with open('temp.txt') as f: 
            assert f.read() == '1\n2\n\n\nthird line'
