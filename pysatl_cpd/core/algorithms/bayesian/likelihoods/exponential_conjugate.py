"""
Module for exponential likelihood function with gamma prior used in Bayesian change point detection. Also contains its'
extension for a sample's probability evaluation with estimated prior parameters.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Optional

import numpy as np
import scipy.stats
from numpy import typing as npt

from pysatl_cpd.core.algorithms.bayesian.abstracts.ilikelihood import (
    ILikelihood,
    ILikelihoodWithPriorProbability,
)


class ExponentialConjugate(ILikelihood):
    """
    Class implementing exponential likelihood function with conjugate gamma prior for Bayesian change point detection.
    Note: it's support is [0; +inf)
    """

    def __init__(self) -> None:
        self._shape_prior: Optional[np.float64] = None
        self._scale_prior: Optional[np.float64] = None

        self.__shapes: npt.NDArray[np.float64] = np.array([])
        self.__scales: npt.NDArray[np.float64] = np.array([])

    def learn(self, learning_sample: npt.NDArray[np.float64]) -> None:
        """
        Learns starting prior parameters to model exponential distribution's likelihood function.
        :param learning_sample: sample to learn starting prior parameters.
        :return:
        """
        self._shape_prior = np.float64(learning_sample.shape[0])
        self._scale_prior = np.sum(learning_sample)

        assert self._shape_prior is not None
        assert self._scale_prior is not None

        self.__shapes = np.array([self._shape_prior])
        self.__scales = np.array([self._scale_prior])

    def update(self, observation: np.float64) -> None:
        """
        Updates parameters (calculating posterior parameters) after a given new observation.
        :param observation: a new observation of time series.
        :return:
        """
        assert self._shape_prior is not None
        assert self._scale_prior is not None

        self.__shapes = np.append([self._shape_prior], (self.__shapes + 1.0))
        self.__scales = np.append([self._scale_prior], (self.__scales + observation))

    def predict(self, observation: np.float64) -> npt.NDArray[np.float64]:
        """
        Calculates predictive posterior probabilities of exponential likelihood for corresponding values of run length.
        :param observation: a new observation of time series.
        :return: an array of predictive posterior probabilities (densities).
        """
        assert self._shape_prior is not None
        assert self._scale_prior is not None

        predictive_probabilities = scipy.stats.lomax.pdf(
            x=observation,
            c=self.__shapes,
            loc=0.0,
            scale=self.__scales,
        )

        # In case of negative scale parameter corresponding distribution does not exist, so substitution of
        # an observation results in a NaN-value. In context of algorithm it can be assumed that this probability is 0.
        without_nans = np.nan_to_num(x=predictive_probabilities, nan=0.0)

        return np.array(without_nans)

    def clear(self) -> None:
        """
        Clears a current state of the likelihood, setting parameters to default init values.
        :return:
        """
        self._shape_prior = None
        self._scale_prior = None

        self.__shapes = np.array([])
        self.__scales = np.array([])


class ExponentialConjugateWithPriorProbability(ExponentialConjugate, ILikelihoodWithPriorProbability):
    """
    Exponential likelihood, supporting a sample's probability evaluation with estimated prior parameters.
    """

    def __init__(self) -> None:
        super().__init__()

    def probability_of_learned_prior(self, sample: npt.NDArray[np.float64]) -> np.float64:
        """
        Evaluates probability of a sample with learned prior parameters of exponential conjugate likelihood.
        :param sample: sample for probability's evaluation.
        :return: probability of a sample with learned prior parameters of exponential conjugate likelihood.
        """
        assert self._shape_prior is not None
        assert self._scale_prior is not None

        probabilities_of_learning_sample = scipy.stats.lomax.pdf(
            x=sample,
            c=self._shape_prior,
            loc=0.0,
            scale=self._scale_prior,
        )

        without_nans = np.nan_to_num(x=probabilities_of_learning_sample, nan=0.0)

        probability_of_learning_sample = np.prod(without_nans)
        return np.float64(probability_of_learning_sample)
