from abc import ABC
from abc import abstractmethod

__all__ = ["Matcher"]


class Matcher(ABC):
    @abstractmethod
    def match(self):
        raise NotImplementedError
