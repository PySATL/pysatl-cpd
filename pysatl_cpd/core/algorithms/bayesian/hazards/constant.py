"""
Module for implementation of Bayesian CPD algorithm constant hazard function corresponding to an exponential
distribution.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.bayesian.abstracts import IHazard

__all__ = ["ConstantHazard"]


class ConstantHazard(IHazard):
    """
    A constant hazard function, corresponding to an exponential distribution with a given rate.
    """

    def __init__(self, rate: float):
        """
        Initializes the constant hazard function with a given rate of an underlying exponential distribution.
        :param rate: rate of an underlying exponential distribution.
        """
        self._rate = np.float64(rate)
        assert self._rate >= 1.0, "Hazard rate cannot be less than 1.0"

    def hazard(self, run_lengths: npt.NDArray[np.intp]) -> npt.NDArray[np.float64]:
        """
        Calculates the constant hazard function.
        :param run_lengths: run lengths at the time.
        :return: hazard function's values for given run lengths.
        """
        return np.ones(len(run_lengths)) / self._rate
