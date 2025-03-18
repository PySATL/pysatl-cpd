"""
Module for online change point detection algorithm's interface.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2025 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Optional, Protocol

import numpy as np
import numpy.typing as npt


class OnlineCpdAlgorithm(Protocol):
    """
    Class for online change point detection algorithm's interface.
    """
    def detect(self, value: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Method for detection of a change point.
        :param value: new value of a time series.
        :return: bool value whether a change point was detected after processing the new value.
        """
        ...

    def localize(self, value: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Method for localization of a change point.
        :param value: new value of a time series
        :return: location of a change point, acquired after processing the new value, or None if there wasn't any.
        """
        ...
