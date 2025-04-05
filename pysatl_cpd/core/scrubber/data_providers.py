"""
Module contains classes providing data from different sources to scrubbers.
"""

__author__ = "Vladimir Kutuev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from pysatl_cpd.labeled_data import LabeledCpdData


@runtime_checkable
class DataProvider(Protocol):
    """Interface for abstracting the scrubber from the data source and its format"""

    def __iter__(self) -> Iterator[np.float64] | Iterator[npt.NDArray[np.float64]]:
        """
        :return: an iterator over the data
        """
        ...


class ListUnivariateProvider(DataProvider):
    """Provides data from list of floats"""

    def __init__(self, data: list[float]) -> None:
        self._data = data

    def __iter__(self) -> Iterator[np.float64] | Iterator[npt.NDArray[np.float64]]:
        return map(np.float64, self._data)


class ListMultivariateProvider(DataProvider):
    """Provides data from list of NumPy ndarrays"""

    def __init__(self, data: list[npt.NDArray[np.float64]]) -> None:
        self._data = data

    def __iter__(self) -> Iterator[np.float64] | Iterator[npt.NDArray[np.float64]]:
        return iter(self._data)


class LabeledDataProvider(DataProvider):
    """Provides data from LabeledData instance"""

    def __init__(self, data: LabeledCpdData) -> None:
        self._data = data.raw_data

    def __iter__(self) -> Iterator[np.float64] | Iterator[npt.NDArray[np.float64]]:
        return iter(self._data)
