from typing import Optional

class TextBasedCommunication:
    def __init__(self, prompt: str = "Enter your message: "):
        self.prompt = prompt

    def setup(self):
        print("Text-based communication is now active.")

    def send_message(self, message: str) -> None:
        print(f"Sending message: {message}")

    def receive_message(self) -> Optional[str]:
        return input(self.prompt)