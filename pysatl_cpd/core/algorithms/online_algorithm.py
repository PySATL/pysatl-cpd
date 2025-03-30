"""
Module for online change point detection algorithm's interface.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Optional, Protocol

import numpy as np
import numpy.typing as npt


class OnlineAlgorithm(Protocol):
    """
    Protocol for online change point detection algorithm's interface.
    """

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Method for a step of detection of a change point.
        :param observation: new observation of a time series.
        :return: bool observation whether a change point was detected after processing the new observation.
        """
        ...

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Method for a step of localization of a change point.
        :param observation: new observation of a time series
        :return: location of a change point, acquired after processing the new observation, or None if there wasn't any.
        """
        ...
