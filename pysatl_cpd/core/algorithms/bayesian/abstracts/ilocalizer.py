"""
Module for Bayesian CPD algorithm localizer's abstract base class.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Protocol

import numpy as np
import numpy.typing as npt


class ILocalizer(Protocol):
    """
    Protocol for localizers that localize a change point with given growth probabilities for run lengths.
    """

    def localize(self, growth_probs: npt.NDArray[np.float64]) -> int:
        """
        Localizes a change point with given growth probabilities for run lengths.
        :param growth_probs: growth probabilities for run lengths at the time.
        :return: run length corresponding with a change point.
        """
        ...
