from abc import ABC, abstractmethod


class Loader(ABC):
    """
    Abstract base class for implementing data loading functionality.

    This class defines a callable interface for loading data from various sources.
    Concrete implementations should provide the actual loading mechanism.
    """

    @abstractmethod
    def __call__(self, data_keys: set[str]) -> dict[str, float]:
        """
        Load and return data from the source.

        :param data_keys: Set containing string keys to load
        :return: Dict with keys from 'data_keys' and values as float from storage
        :raises NotImplementedError: If not implemented in subclass

        .. rubric:: Example Return Format

        Typical returned data structure::

           {"temperature": 23.5, "pressure": 1013.2, "humidity": 45.0}
        """
        ...
