from abc import ABC, abstractmethod


class Saver(ABC):
    """
    Abstract base class for implementing data saving functionality.

    This class provides an interface for saving dictionary data to various storage systems.
    Concrete implementations should define the actual saving mechanism in the __call__ method.
    """

    @abstractmethod
    def __call__(self, data: dict[str, float]) -> None:
        """Save the provided data to the storage system.

        :param data: Dictionary containing key-value pairs to be saved.
                    Keys are strings, values are floating-point numbers.
        :raises NotImplementedError: If not implemented in subclass
        :return: None

        .. rubric:: Example Data Format

        Typical input data might look like::

           {"temperature": 23.5, "pressure": 1013.2, "humidity": 45.0}
        """
        ...
