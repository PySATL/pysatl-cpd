from abc import ABC, abstractmethod
from typing import Any


class Loader(ABC):
    """
    Abstract base class for implementing data loading functionality.

    This class defines a callable interface for loading data from various sources.
    Concrete implementations should provide the actual loading mechanism.
    """

    @abstractmethod
    def __call__(self, data_keys: set[str]) -> dict[str, dict[Any, Any]]:
        """
        Load and return data from the source.

        :param data_keys: Set containing string keys to load
        :return: Dict with keys from 'data_keys' and values as dicts
        :raises NotImplementedError: If not implemented in subclass

        .. rubric:: Example Return Format

        Typical returned data structure::

           {"temperature": {1: 32.3, 2: 33.4}, "pressure": {1: 1013, 2: 1200}}
        """
        ...
