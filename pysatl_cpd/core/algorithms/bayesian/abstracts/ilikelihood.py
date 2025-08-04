"""
Module for Bayesian CPD algorithm likelihood function's abstract base class and its' extension for a sample's
probability evaluation with estimated prior parameters.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Protocol

import numpy as np
import numpy.typing as npt

__all__ = ["ILikelihood", "ILikelihoodWithPriorProbability"]


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


class ILikelihoodWithPriorProbability(ILikelihood, Protocol):
    """
    Likelihood which also allows to evaluate how probable is learning sample with learned prior parameters.
    """

    def probability_of_learned_prior(self, sample: npt.NDArray[np.float64]) -> np.float64:
        """
        Evaluation of how probable is learning sample with learned prior parameters.
        :param sample: a sample for the likelihood.
        :return: probability of getting a learning sample with learned prior parameters.
        """
        ...
