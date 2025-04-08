"""
Module for Bayesian CPD algorithm detector's abstract base class.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Protocol

import numpy as np
import numpy.typing as npt


class IDetector(Protocol):
    """
    Protocol for detectors that detect a change point with given growth probabilities for run lengths.
    """

    def detect(self, growth_probs: npt.NDArray[np.float64]) -> bool:
        """
        Checks whether a changepoint occurred with given growth probabilities at the time.
        :param growth_probs: growth probabilities for run lengths at the time.
        :return: boolean indicating whether a changepoint occurred
        """
        ...

    def clear(self) -> None:
        """
        Clears the detector's state.
        """
        ...
