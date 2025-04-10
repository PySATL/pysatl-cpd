"""
Module for exponential likelihood function with gamma prior used in Bayesian change point detection.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import scipy.stats
from numpy import typing as npt

from pysatl_cpd.core.algorithms.bayesian.abstracts.ilikelihood import ILikelihood


class ExponentialConjugate(ILikelihood):
    """
    Class implementing exponential likelihood function with conjugate gamma prior for Bayesian change point detection.
    Note: it's support is [0; +inf)
    """

    def __init__(self) -> None:
        self.__shape_prior: np.float64 | None = None
        self.__scale_prior: np.float64 | None = None

        self.__shapes: npt.NDArray[np.float64] = np.array([])
        self.__scales: npt.NDArray[np.float64] = np.array([])

    def learn(self, learning_sample: npt.NDArray[np.float64]) -> None:
        """
        Learns starting prior parameters to model exponential distribution's likelihood function.
        :param learning_sample: sample to learn starting prior parameters.
        :return:
        """
        self.__shape_prior = np.float64(learning_sample.shape[0])
        self.__scale_prior = np.sum(learning_sample)

        assert self.__shape_prior is not None
        assert self.__scale_prior is not None

        self.__shapes = np.array([self.__shape_prior])
        self.__scales = np.array([self.__scale_prior])

    def update(self, observation: np.float64) -> None:
        """
        Updates parameters (calculating posterior parameters) after a given new observation.
        :param observation: a new observation of time series.
        :return:
        """
        assert self.__shape_prior is not None
        assert self.__scale_prior is not None

        self.__shapes = np.append([self.__shape_prior], (self.__shapes + 1.0))
        self.__scales = np.append([self.__scale_prior], (self.__scales + observation))

    def predict(self, observation: np.float64) -> npt.NDArray[np.float64]:
        """
        Calculates predictive posterior probabilities of exponential likelihood for corresponding values of run length.
        :param observation: a new observation of time series.
        :return: array of predictive posterior probabilities (densities).
        """
        assert self.__shape_prior is not None
        assert self.__scale_prior is not None

        predictive_probabilities = scipy.stats.lomax.pdf(
            x=observation,
            c=self.__shapes,
            loc=0.0,
            scale=self.__scales,
        )

        without_nans = np.nan_to_num(x=predictive_probabilities, nan=0.0)

        return np.array(without_nans)

    def clear(self) -> None:
        """
        Clears a current state of the likelihood, setting parameters to default init values.
        :return:
        """
        self.__shape_prior = None
        self.__scale_prior = None

        self.__shapes = np.array([])
        self.__scales = np.array([])
