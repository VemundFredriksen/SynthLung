from abc import abstractmethod, ABC

class JSONGenerator(ABC):

    @abstractmethod
    def generate_json(self) -> None:
        pass