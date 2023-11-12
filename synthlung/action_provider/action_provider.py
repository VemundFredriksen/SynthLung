from abc import abstractmethod, ABC

class ActionProvider(ABC):
    
    @abstractmethod
    def start_new_session(self) -> None:
        pass

    @abstractmethod
    def get_session_id(self) -> None:
        pass

    @abstractmethod
    def get_config(self) -> None:
        pass

    @abstractmethod
    def save_log(self, log_dict) -> None:
        pass