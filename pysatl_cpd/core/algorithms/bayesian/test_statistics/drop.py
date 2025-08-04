"""
Module for implementation of Bayesian CPD algorithm detector analyzing drop of maximal run length's probability.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.bayesian.abstracts import IChangePointFunction

__all__ = ["DropCPF"]


class DropCPF(IChangePointFunction):
    """
    A change point function based on the instantaneous drop in the probability
    of the maximum run length.
    """

    def __init__(self):
        """
        Initializes the DropCPF.
        """
        self.__previous_growth_prob: Optional[float] = None

    def calculate(self, growth_probs: npt.NDArray[np.float64]) -> float:
        """
        Checks whether a changepoint occurred with given growth probabilities at the time.
        :param growth_probs: growth probabilities for run lengths at the time.
        :return: boolean indicating whether a changepoint occurred.
        """
        if len(growth_probs) == 0:
            return 0.0

        last_growth_prob = growth_probs[-1]
        if self.__previous_growth_prob is None:
            self.__previous_growth_prob = last_growth_prob
            return 0.0

        drop = float(self.__previous_growth_prob - last_growth_prob)
        return drop

    def clear(self) -> None:
        """
        Clears the detector's state.
        """
        self.__previous_growth_prob = None
