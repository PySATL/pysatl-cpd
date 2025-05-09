from abc import ABC, abstractmethod


class Saver(ABC):
    @abstractmethod
    def __call__(self, data: dict[str, float]) -> None:
        ...
