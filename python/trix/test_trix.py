import numpy as np
from numpy.testing import assert_equal
from nose.tools import raises
from trix import check_orthonormal, plot_rv_cont, replicate


class test_check_orthonormal:

    @raises(ValueError)
    def test_ndarray_fail(self):
        check_orthonormal(np.random.randn(3, 3, 3))


class test_plot_rv_cont:
    pass


class test_replicate:

    def flagger(self, flag=True):
        '''
        Dummy function for testing here
        '''
        return max(flag, 0)

    def multi(self):
        
        return (0, 0)

    def test_works_with_no_args(self):

        actual = replicate(self.flagger, 5)
        assert_equal(actual, np.ones(5))

    def test_works_with_keyword_args(self):

        actual = replicate(self.flagger, 5, flag=False)
        assert_equal(actual, np.zeros(5))

    def test_handles_vectorized_return(self):
        actual = replicate(self.multi, 5)
        assert_equal(actual, np.zeros((5, 2)))
