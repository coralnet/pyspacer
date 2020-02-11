import unittest

from spacer import mailman


class TestMailman(unittest.TestCase):

    def test_handle_message_input_type(self):
        self.assertRaises(AttributeError, mailman.handle_message, 'sdf')

    def setUp(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
