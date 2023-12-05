import unittest

from PIL import Image

from spacer.exceptions import RowColumnInvalidError
from spacer.task_utils import check_extract_inputs


class TestRowColChecks(unittest.TestCase):

    def test_ints(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(1.1, 1.2)]
        with self.assertRaises(RowColumnInvalidError):
            check_extract_inputs(img, rowcols, 'img')

    def test_ok(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(0, 0), (99, 99)]
        try:
            check_extract_inputs(img, rowcols, 'img')
        except AssertionError:
            self.fail("check_extract_inputs raised error unexpectedly")

    def test_negative(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(-1, -1)]
        with self.assertRaises(RowColumnInvalidError):
            check_extract_inputs(img, rowcols, 'img')

    def test_too_large(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(100, 100)]
        with self.assertRaises(RowColumnInvalidError):
            check_extract_inputs(img, rowcols, 'img')

    def test_row_too_large(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(100, 99)]

        with self.assertRaises(RowColumnInvalidError) as context:
            check_extract_inputs(img, rowcols, 'img')
        self.assertEqual(
            str(context.exception),
            "img: Row value 100 falls outside this image's"
            " valid range of 0-99.")

    def test_col_too_large(self):
        img = Image.new('RGB', (100, 100))
        rowcols = [(99, 100)]

        with self.assertRaises(RowColumnInvalidError) as context:
            check_extract_inputs(img, rowcols, 'img')
        self.assertEqual(
            str(context.exception),
            "img: Column value 100 falls outside this image's"
            " valid range of 0-99.")


if __name__ == '__main__':
    unittest.main()
