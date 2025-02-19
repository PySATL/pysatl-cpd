"""
Module for Abstract Scrubber description.
"""

__author__ = "Romanyuk Artem, Vladimir Kutuev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.scrubber.data_providers import DataProvider


@dataclass
class ScrubberWindow:
    values: npt.NDArray[np.float64]
    indices: list[int]


class Scrubber(ABC):
    """A scrubber for dividing data into windows
    and subsequent processing of data windows
    by change point detection algorithms
    """

    def __init__(self, data_provider: DataProvider) -> None:
        """A scrubber for dividing data into windows
        and subsequent processing of data windows
        by change point detection algorithms

        """
        self._data_provider = data_provider

    @abstractmethod
    def __iter__(self) -> Iterator[ScrubberWindow]:
        """Function for dividing data into parts to feed into the change point detection algorithm

        :return: Iterator of data windows for change point detection algorithm
        """
        ...

    @property
    def data(self) -> Iterator[np.float64] | Iterator[npt.NDArray[np.float64]]:
        return iter(self._data_provider)
