"""
Module for Bayesian CPD algorithm hazard function's abstract base class.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class IHazard(ABC):
    """
    Hazard function abstract base class.
    """

    @abstractmethod
    def hazard(self, run_lengths: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculates the hazard function for given run lengths.
        :param run_lengths: run lengths at the time.
        :return: hazard function's values for given run lengths.
        """
        raise NotImplementedError
