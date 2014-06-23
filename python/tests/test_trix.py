import numpy as np
from nose.tools import raises
from trix import check_orthonormal


class test_check_orthonormal:

    @raises(ValueError)
    def test_ndarray_fail(self):
        check_orthonormal(np.random.randn(3, 3, 3))
