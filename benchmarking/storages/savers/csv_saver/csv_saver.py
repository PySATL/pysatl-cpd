"""
Module, that implements Saver, based of CSV.

SaverCSV saves data into CSV files.
"""

__author__ = "Aleksei Ivanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import csv
from pathlib import Path

import numpy as np

from benchmarking.custom_types import StorageValues
from benchmarking.storages.savers.saver import Saver


class SaverCSV(Saver):
    """CSV data saver"""

    def __init__(self, step_storages_name: str = "generation"):
        self.directory = Path("experiment_storages") / "csv" / step_storages_name
        self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, storage_name: str, data: StorageValues) -> None:
        """Saves data to experiment_storage/[step_storage_name]/[storage_name].csv with key,value from given dict"""
        if isinstance(data, float | int | str):
            storage_name += "_literal"
            new_data = {0: data}
        elif isinstance(data, list | tuple | np.ndarray):
            storage_name += "_list"
            new_data = {}
            for i in range(len(data)):
                new_data[i] = data[i]
        elif isinstance(data, dict):
            new_data = data
        else:
            raise TypeError(f"wrong data type ({type(data)})")

        filename = self.directory / f"{storage_name}.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])

            for key, value in new_data.items():
                writer.writerow([key, value])
