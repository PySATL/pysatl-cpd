"""
Module for implementation of Linear Scrubber.
"""

__author__ = "Romanyuk Artem"
__copyright__ = "Copyright (c) 2024 Romanyuk Artem"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterator
from itertools import islice

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.scrubber.data_providers import DataProvider

from .abstract import Scrubber, ScrubberWindow


class LinearScrubber(Scrubber):
    """A linear scrubber for dividing data into windows by moving them through data"""

    def __init__(
        self,
        data_provider: DataProvider,
        window_length: int = 100,
        shift_factor: float = 1.0 / 3.0,
    ):
        """A linear scrubber for dividing data into windows by moving them through data

        :param window_length: length of data window
        :param shift_factor: how far will the window move relative to the length
        """
        super().__init__(data_provider)
        self._window_length = window_length
        self._shift_factor = shift_factor
        self._rewrite_data_index: int = 0

    def __iter__(self) -> Iterator[ScrubberWindow]:
        window_start = 0
        shift = max(1, int(self._window_length * self._shift_factor))
        provided_data_it = iter(self._data_provider)
        next_slice = np.array(list(islice(provided_data_it, self._window_length)))
        window_data: npt.NDArray[np.float64] = np.array([])
        while next_slice.size > 0:
            # print(np.delete(window_data, np.s_[:shift], 0))
            # print(next_slice)
            window_data = np.concat((np.delete(window_data, np.s_[:shift], 0), next_slice), axis=0) if len(window_data) > 0 else next_slice
            window_end = window_start + min(self._window_length, len(window_data))
            yield ScrubberWindow(window_data, list(range(window_start, window_end)))
            window_start += shift
            window_end += shift
            next_slice = np.array(list(islice(provided_data_it, shift)))
