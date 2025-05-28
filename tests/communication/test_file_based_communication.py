import unittest
from communication.file_based_communication import FileBasedCommunication

class TestFileBasedCommunication(unittest.TestCase):
    def setUp(self):
        self.comm = FileBasedCommunication(file_path="test_messages.txt")

    def test_send_message(self):
        result = self.comm.send_message("Hello, World!")
        self.assertTrue(result)

    def test_receive_message(self):
        message = "Hello, World!"
        self.comm.send_message(message)
        received_message = self.comm.receive_message()
        self.assertEqual(received_message, message)

if __name__ == '__main__':
    unittest.main()