import subprocess
import os

from nose.tools import assert_equal


# Go two directories up in bin to get rstrip
fpath = os.path.dirname(__file__)
rs = os.sep.join([fpath, os.pardir, os.pardir, 'bin', 'rstrip'])


class test_rstrip:

    def setup(self):
        with open('temp.txt', 'w') as f:
            f.write('  1    \n   2  \n\n\nthird line    \n')

        subprocess.call([rs, 'temp.txt'])

    def teardown(self):
        os.remove('temp.txt')

    def test_simple_rstrip(self):
        with open('temp.txt') as f:
            assert_equal(f.read(), '  1\n   2\n\n\nthird line\n')

    def test_removes_backup_file(self):
        assert not os.path.isfile('temp.txt.bak')
