import unittest
from PIL import Image
from spacer.task_utils import check_rowcols


class TestRowColCheck(unittest.TestCase):

    def test_ints(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(1.1, 1.2)]
        self.assertRaises(AssertionError, check_rowcols, rowcols, img)

    def test_ok(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(0, 0), (99, 99)]
        try:
            check_rowcols(rowcols, img)
        except AssertionError:
            self.fail("check_rowcols raised error unexpectedly")

    def test_negative(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(-1, -1)]
        self.assertRaises(AssertionError, check_rowcols, rowcols, img)

    def test_too_large(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(100, 100)]
        self.assertRaises(AssertionError, check_rowcols, rowcols, img)

    def test_row_too_large(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(100, 99)]
        try:
            check_rowcols(rowcols, img)
        except AssertionError as err:
            self.assertIn('row', repr(err))
            self.assertNotIn('col', repr(err))

    def test_col_too_large(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(99, 100)]
        try:
            check_rowcols(rowcols, img)
        except AssertionError as err:
            self.assertIn('col', repr(err))
            self.assertNotIn('row', repr(err))
