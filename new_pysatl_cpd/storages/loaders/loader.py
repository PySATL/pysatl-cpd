"""
Module, that defines an interface of Loader class.

Loader class performs fetching data from the storage.
Type of storage and the way of fetching depends on concrete realization.
"""

__author__ = "Aleksei Ivanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from new_pysatl_cpd.custom_types import StorageValues


class Loader(ABC):
    """
    Abstract base class for implementing data loading functionality.

    This class defines a callable interface for loading data from various sources.
    Concrete implementations should provide the actual loading mechanism.
    """

    @abstractmethod
    def __call__(self, data_keys: set[str]) -> dict[str, StorageValues]:
        """
        Load and return data from the source.

        :param data_keys: Set containing string keys to load
        :return: Dict with keys from 'data_keys' and values as StorageValues
        :raises NotImplementedError: If not implemented in subclass

        .. rubric:: Example Return Format

        Typical returned data structure::

           {"temperature": {1: 32.3, 2: 33.4}, "pressure": {1: 1013, 2: 1200}} or
           {"temperature": [32.3, 33.4], "pressure": [1013, 1200]}
        """
        ...
