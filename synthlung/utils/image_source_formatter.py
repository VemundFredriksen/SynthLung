from abc import abstractmethod, ABC

class ImageSourceFormatter(ABC):

    @abstractmethod
    def format(self) -> None:
        pass