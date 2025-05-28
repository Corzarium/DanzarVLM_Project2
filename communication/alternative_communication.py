from abc import ABC, abstractmethod

class AlternativeCommunication(ABC):
    @abstractmethod
    def send_message(self, message: str) -> None:
        pass

    @abstractmethod
    def receive_message(self) -> str:
        pass

    @abstractmethod
    def setup(self) -> None:
        pass