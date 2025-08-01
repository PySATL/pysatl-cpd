"""
Module for implementation of Bayesian CPD algorithm detector analyzing drop of maximal run length's probability.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.bayesian.abstracts import IDetector


class DropDetector(IDetector):
    """
    A detector that detects a change point if the instantaneous drop in the probability of the maximum run length
    exceeds the threshold.
    """

    def __init__(self, threshold: float):
        """
        Initializes the detector with given drop threshold.
        :param threshold: threshold for a drop of the maximum run length's probability.
        """
        self.__previous_growth_prob: Optional[float] = None

        self._threshold = threshold
        assert 0.0 <= self._threshold <= 1.0, "Drop threshold must be in [0.0, 1.0]"

    def detect(self, growth_probs: npt.NDArray[np.float64]) -> bool:
        """
        Checks whether a changepoint occurred with given growth probabilities at the time.
        :param growth_probs: growth probabilities for run lengths at the time.
        :return: boolean indicating whether a changepoint occurred.
        """
        if len(growth_probs) == 0:
            return False

        last_growth_prob = growth_probs[-1]
        if self.__previous_growth_prob is None:
            self.__previous_growth_prob = last_growth_prob
            return False

        drop = float(self.__previous_growth_prob - last_growth_prob)

        return drop >= self._threshold

    def clear(self) -> None:
        """
        Clears the detector's state.
        """
        self.__previous_growth_prob = None
