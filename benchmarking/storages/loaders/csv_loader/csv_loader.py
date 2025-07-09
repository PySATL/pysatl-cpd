"""
Module, that implements a Loader, based of CSV.

LoaderCSV loads data from CSV files.
"""

__author__ = "Aleksei Ivanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import csv
from pathlib import Path
from typing import Optional, Union, cast

from benchmarking.custom_types import StorageValues
from benchmarking.storages.loaders.loader import Loader


class LoaderCSV(Loader):
    """CSV data loader"""

    def __init__(self, step_storages_name: str = "generation"):
        self.directory = Path("experiment_storages") / "csv" / step_storages_name

    def __call__(self, data_keys: set[str]) -> dict[str, StorageValues]:
        result = {}
        for key in data_keys:
            loaded_data = self._load_key_data(key)
            if loaded_data is not None:
                result[key] = loaded_data
        return result

    def _load_key_data(self, key: str) -> Optional[StorageValues]:
        filename = self._find_existing_file(key)
        if not filename:
            return None

        file_data = self._read_file_data(filename)
        if filename.name.endswith("_literal.csv"):
            return file_data.get("0")
        if filename.name.endswith("_list.csv"):
            return self._convert_to_list(file_data)
        return file_data

    def _find_existing_file(self, key: str) -> Optional[Path]:
        for suffix in ["", "_list", "_literal"]:
            filename = self.directory / f"{key}{suffix}.csv"
            if filename.exists():
                return filename
        return None

    def _read_file_data(self, filename: Path) -> dict[str, Union[int, float, str]]:
        file_data = {}
        NUM_OF_COLS = 2
        with open(filename) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < NUM_OF_COLS:
                    continue
                file_data[row[0]] = self._convert_value(row[1])
        return file_data

    def _convert_value(self, value: str) -> Union[int, float, str]:
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    def _convert_to_list(self, file_data: dict[str, Union[int, float, str]]) -> list[str] | list[int] | list[float]:
        result_list = list(file_data.values())
        first_element = result_list[0] if result_list else None
        if self._is_int(first_element):
            return cast(list[int], result_list)
        if self._is_float(first_element):
            return cast(list[float], result_list)
        return cast(list[str], result_list)

    @staticmethod
    def _is_float(element: str) -> bool:
        try:
            float(element)
            return True
        except Exception:
            return False

    @staticmethod
    def _is_int(element: str) -> bool:
        try:
            int(element)
            return True
        except Exception:
            return False
