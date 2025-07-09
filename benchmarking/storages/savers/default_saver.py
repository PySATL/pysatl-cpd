"""
Module contains dummy realization of saver for testing and debugging.

works by saving all data in a field inside the object.
"""

__author__ = "Aleksei Ivanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from benchmarking.custom_types import StorageValues
from benchmarking.logger import cpd_logger
from benchmarking.storages.savers.saver import Saver


class DefaultSaver(Saver):
    """Dummy saver without realisation"""

    def __init__(self, dict_as_db: dict[str, StorageValues]):
        self.dict_as_db = dict_as_db

    def __call__(self, storage_name: str, data: StorageValues) -> None:
        cpd_logger.info(f"Saved: {storage_name}")
        self.dict_as_db[storage_name] = data
