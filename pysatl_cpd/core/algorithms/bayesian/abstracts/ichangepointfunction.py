"""
Module for Bayesian CPD algorithm detector's abstract base class.
"""

__author__ = "Alexey Tatyanenko, Vladimir Kutuev"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Protocol

import numpy as np
import numpy.typing as npt

__all__ = ["IChangePointFunction"]


class IChangePointFunction(Protocol):
    """
    Protocol for test_statistics that detect a change point with given growth probabilities for run lengths.
    """

    def calculate(self, growth_probs: npt.NDArray[np.float64]) -> float:
        """Calculates change point function using given growth probabilities.

        :param growth_probs: Growth probabilities for run lengths.
        :return: Change point function value.
        """
        ...

    def clear(self) -> None:
        """Clears the change point function's internal state."""
        ...
