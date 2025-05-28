from communication.abstract_communication import AbstractCommunication

class FileBasedCommunication(AbstractCommunication):
    def __init__(self, file_path):
        self.file_path = file_path
        super().__init__()

    def setup(self):
        # Initialize any necessary resources for communication
        pass

    def send_message(self, message):
        try:
            with open(self.file_path, 'a') as file:
                file.write(message + '\n')
            return True
        except Exception as e:
            print(f"Error sending message: {e}")
            return False

    def receive_message(self):
        messages = self.receive_messages()
        if messages:
            return messages.pop(0).strip()
        return ""

    def receive_messages(self):
        try:
            with open(self.file_path, 'r') as file:
                return [line.strip() for line in file.readlines()]
        except Exception as e:
            print(f"Error receiving messages: {e}")
            return []