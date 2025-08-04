"""
Module for implementation of Bayesian CPD algorithm detector comparing maximal run length's probability with
a threshold.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.bayesian.abstracts.ichangepointfunction import IChangePointFunction

__all__ = ["MaxRunLengthCPF"]


class MaxRunLengthCPF(IChangePointFunction):
    """
    Change point function based on the probability of the maximum run length.
    """

    def calculate(self, growth_probs: npt.NDArray[np.float64]) -> float:
        if len(growth_probs) == 0:
            return 0.0
        return 1.0 - float(growth_probs[-1])

    def clear(self) -> None:
        return
