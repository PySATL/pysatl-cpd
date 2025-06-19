"""
Module contains dummy realization of loader for testing and debugging.

It works by getting data from class object, that was previously saved.
"""

__author__ = "Aleksei Ivanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from new_pysatl_cpd.custom_types import StorageValues
from new_pysatl_cpd.storages.loaders.loader import Loader


class DefaultLoader(Loader):
    """Dummy loader without realisation"""

    def __init__(self, dict_as_db: dict[str, StorageValues]):
        self.dict_as_db = dict_as_db

    def __call__(self, data_keys: set[str]) -> dict[str, StorageValues]:
        result = {}
        for key in data_keys:
            result[key] = self.dict_as_db[key]
        return result
