import numpy as np
from numpy.testing import assert_equal, raises, assert_raises_regex
from trix import check_orthonormal, replicate, bootstrap


class test_check_orthonormal:

    @raises(ValueError)
    def test_ndarray_fail(self):
        check_orthonormal(np.random.randn(3, 3, 3))


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


class test_bootstrap():

    def test_iter_value(self):
        actual = bootstrap(np.ones(5), reps=3, lazy=True)
        assert_equal(1.0, next(actual))

    def test_iter_decrements_reps(self):
        actual = bootstrap(np.ones(5), reps=50, lazy=True)
        next(actual)
        assert_equal(49, actual._reps_remain)

    def test_list_comprehension(self):
        b = bootstrap(np.ones(5), np.mean, reps=5, lazy=True)
        actual = [x for x in b]
        assert_equal([1] * 5, actual)

    def test_len(self):
        actual = bootstrap(np.ones(5), reps=5)
        assert_equal(5, len(actual))

    def test_repr(self):
        r = repr(bootstrap(np.ones(5), stat=np.mean, reps=5))
        for string in ['stat', 'mean', 'reps', '5']:
            assert string in r

    @raises(StopIteration)
    def test_doesnt_iterate_infinitely(self):
        b = bootstrap(np.ones(5))
        next(b)

    def test_results_attribute_available_after_run(self):
        actual = bootstrap(np.ones(5), reps=10)
        assert_equal(np.ones(10), actual.results)

    def test_cant_compute_standard_error_without_results(self):
        actual = bootstrap(np.ones(5), lazy=True)
        assert_raises_regex(AttributeError, 'stderror', actual.stderror)

    def test_cant_compute_confidence_without_results(self):
        actual = bootstrap(np.ones(5), lazy=True)
        assert_raises_regex(AttributeError, 'confidence', actual.confidence)

    def test_actual_available(self):
        b = bootstrap(np.ones(5), reps=3)
        assert_equal(b.actual, 1)

    def test_sorts_results_array(self):
        np.random.seed(20)
        b = bootstrap(np.arange(20), reps=10)
        assert_equal(b.results, sorted(b.results))
