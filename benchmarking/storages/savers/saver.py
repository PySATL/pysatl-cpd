"""
Module, that defines an interface of Saver class.

Saver class performs saving data into the storage.
Type of storage and the way of storing depends on concrete realization.
"""

__author__ = "Aleksei Ivanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from benchmarking.custom_types import StorageValues


class Saver(ABC):
    """
    Abstract base class for implementing data saving functionality.

    This class provides an interface for saving dictionary data to various storage systems.
    Concrete implementations should define the actual saving mechanism in the __call__ method.
    """

    @abstractmethod
    def __call__(self, storage_name: str, data: StorageValues) -> None:
        """Save the provided data to the storage system.

        :param storage_name: The name under which the data will be saved.
                    In later steps data can be achieved by this name.
        :param data: :param data: data to be saved. Can be int, str or list[int], list[str], dict[Any, Any].
                    Every type should be treated differently.
        :raises NotImplementedError: If not implemented in subclass
        :return: None

        .. rubric:: Example Data Format

        Typical input data might look like::

           "temperature", {1: 32.3, 2: 33.4} or
           "temperature", [32.3, 33.4] or
           "accuracy", 0.994 or
           "weather", "humid"
        """
        ...
