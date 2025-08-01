"""
Module for implementation of Bayesian CPD algorithm detector comparing maximal run length's probability with
a threshold.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.bayesian.abstracts import IDetector


class ThresholdDetector(IDetector):
    """
    A detector that detects a change point if the probability of the maximum run length drops below the threshold.
    """

    def __init__(self, threshold: float):
        """
        Detects a change point if the probability of the maximum run length drops below the threshold.
        :param threshold: lower threshold for the maximum run length's probability.
        """
        self._threshold = threshold
        assert 0.0 <= self._threshold <= 1.0, "Threshold must be in [0.0, 1.0]"

    def detect(self, growth_probs: npt.NDArray[np.float64]) -> bool:
        """
        Detects a change point if the probability of the maximum run length drops below the threshold.
        :param growth_probs: growth probabilities for run lengths at the time.
        :return: boolean indicating whether a changepoint occurred.
        """
        return len(growth_probs) > 0 and growth_probs[-1] < self._threshold

    def clear(self) -> None:
        """
        Clears the detector's state (for this detector it does nothing).
        """
        pass
