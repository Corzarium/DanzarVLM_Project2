import requests

class HttpCommunication:
    def __init__(self, url):
        self.url = url

    def setup(self):
        # Any setup logic if needed
        pass

    def send_message(self, message):
        response = requests.post(self.url, json={"message": message})
        return response.json().get("response", "")

    def receive_message(self):
        # For simplicity, we'll assume the LLM sends a message to us via HTTP GET
        response = requests.get(self.url)
        return response.json().get("response", "")