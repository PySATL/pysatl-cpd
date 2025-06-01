from abc import ABC, abstractmethod
from typing import Any


class Saver(ABC):
    """
    Abstract base class for implementing data saving functionality.

    This class provides an interface for saving dictionary data to various storage systems.
    Concrete implementations should define the actual saving mechanism in the __call__ method.
    """

    @abstractmethod
    def __call__(self, storage_name: str, data: dict[Any, Any]) -> None:
        """Save the provided data to the storage system.

        :param storage_name: The name under which the data will be saved.
                    In later steps data can be achieved by this name.
        :param data: Dictionary containing key-value pairs to be saved.
                    Keys are strings, values are floating-point numbers.
        :raises NotImplementedError: If not implemented in subclass
        :return: None

        .. rubric:: Example Data Format

        Typical input data might look like::

           "temperature", {1: 32.3, 2: 33.4}
        """
        ...
