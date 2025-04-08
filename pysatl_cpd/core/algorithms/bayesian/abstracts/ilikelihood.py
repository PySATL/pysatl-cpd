"""
Module for Bayesian CPD algorithm likelihood function's abstract base class.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Protocol

import numpy as np
import numpy.typing as npt


class ILikelihood(Protocol):
    """
    Likelihood function's protocol.
    """

    def learn(self, learning_sample: npt.NDArray[np.float64]) -> None:
        """
        Learns first parameters of a likelihood function on a given sample.
        :param learning_sample: a sample for parameter learning.
        """
        ...

    def predict(self, observation: np.float64) -> npt.NDArray[np.float64]:
        """
        Returns predictive probabilities for a given observation based on stored parameters.
        :param observation: an observation from a sample.
        :return: predictive probabilities for a given observation.
        """
        ...

    def update(self, observation: np.float64) -> None:
        """
        Updates parameters of a likelihood function according to the given observation.
        :param observation: an observation from a sample.
        """
        ...

    def clear(self) -> None:
        """
        Clears likelihood function's state.
        """
        ...
