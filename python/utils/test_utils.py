import os
import nose
from nose.tools import assert_equal
from utils import weighcount, flatten, to_csv, search_replace, rstrip


class test_replicate:

    def test_exactly_n(self):
        pass


class test_weighcount:

    def setup(self):
        self.wc = weighcount({'a': 2, 'b': 8, 'c': 10})

    def test_total(self):
        assert_equal(self.wc.total(), 20)

    def test_single_weight(self):
        assert_equal(self.wc.weight('a'), 0.1)

    def test_common_weights_all(self):
        all_weights = [('c', 0.5), ('b', 0.4), ('a', 0.1)]
        assert_equal(self.wc.common_weights(), all_weights)

    def test_common_weights_n(self):
        assert_equal(self.wc.common_weights(2), [('c', 0.5), ('b', 0.4)])


class test_to_csv:

    def setup(self):
        self.no_header = 'a,0\nb,1\nc,2\nd,3\n'
        self.content = zip('abcd', range(4))

    def teardown(self):
        try:
            os.remove('temp.csv')
        except FileNotFoundError:
            pass

    def test_defaults(self):
        to_csv('temp.csv', self.content)
        with open('temp.csv') as f:
            assert_equal(f.read(), self.no_header)

    def test_header(self):
        to_csv('temp.csv', self.content, header=['first', 'second'])
        with open('temp.csv') as f:
            assert_equal(f.read(), 'first,second\n' + self.no_header)

    @nose.tools.raises(ValueError)
    def test_same_number_fields_as_header(self):
        to_csv('temp.csv', self.content, header=['a', 'b', 'c'])

    @nose.tools.raises(ValueError)
    def test_dont_take_length_of_string(self):
        to_csv('temp.csv', self.content, header='ab')


class test_flatten:

    def test_mixed_containers(self):
        mixed = [1, 2, (3, 4, {5})]
        assert_equal(list(flatten(mixed)), [1, 2, 3, 4, 5])

    def test_default_dont_iterate_string(self):
        stringlist = ['abc', 'def']
        assert_equal(list(flatten(stringlist)), ['abc', 'def'])


class test_search_replace:

    def setup(self):
        with open('testfile.txt', 'w') as f:
            f.write('old text\nmore oldy\nand then')

    def teardown(self):
        os.remove('testfile.txt')

    def test_old_changes_to_new(self):
        search_replace('testfile.txt', 'old', 'new')
        with open('testfile.txt') as f:
            assert_equal(f.read(), 'new text\nmore newy\nand then')

    def test_backup_file_is_removed(self):
        search_replace('testfile.txt', 'old', 'new')
        nose.tools.assert_false(os.path.isfile('testfile.txt.bak'))

    def test_silently_skips_directories(self):
        os.mkdir('tempdir')
        search_replace('tempdir', 'old', 'new')
        os.rmdir('tempdir')


class test_rstrip:

    def setup(self):
        with open('somefile.txt', 'w') as f:
            f.writelines('\n'.join(['   first  ', '  ', '  third  ']))

    def teardown(self):
        os.remove('somefile.txt')

    def test_basic_rstrip(self):
        rstrip('somefile.txt')
        with open('somefile.txt') as f:
            assert_equal(f.read(), '   first\n\n  third')
