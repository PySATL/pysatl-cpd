"""
Module for prediction model for Bayesian online CPD, which supports heuristic selection of gaussian (normal) or
exponential conjugate likelihood based on estimation from learning sample.
"""

from typing import Optional

import numpy as np
from numpy import typing as npt

from pysatl_cpd.core.algorithms.bayesian.abstracts.ilikelihood import ILikelihood, ILikelihoodWithPriorProbability
from pysatl_cpd.core.algorithms.bayesian.likelihoods.exponential_conjugate import (
    ExponentialConjugateWithPriorProbability,
)
from pysatl_cpd.core.algorithms.bayesian.likelihoods.gaussian_conjugate import GaussianConjugateWithPriorProbability


class HeuristicGaussianVsExponential(ILikelihood):
    """
    Prediction model class with heuristic selection of gaussian (normal) or exponential conjugate likelihood based on
    estimation from learning sample.
    """

    def __init__(self) -> None:
        self.__likelihood: Optional[ILikelihoodWithPriorProbability] = None
        self.likelihoods_history = []

    def learn(self, learning_sample: npt.NDArray[np.float64]) -> None:
        """
        Learns prior parameters for gaussian and exponential likelihoods, evaluates which makes a learning sample more
        probable and saves acquired likelihood for further work.
        :param learning_sample: a sample to estimate prior parameters and compare likelihoods.
        :return:
        """
        gaussian = GaussianConjugateWithPriorProbability()
        exponential = ExponentialConjugateWithPriorProbability()

        gaussian.learn(learning_sample)
        exponential.learn(learning_sample)

        gaussian_probability = gaussian.probability_of_learned_prior(learning_sample)
        exponential_probability = exponential.probability_of_learned_prior(learning_sample)

        self.__likelihood = gaussian if gaussian_probability >= exponential_probability else exponential
        self.likelihoods_history.append("N" if gaussian_probability >= exponential_probability else "E")

    def predict(self, observation: np.float64) -> npt.NDArray[np.float64]:
        """
        Returns prediction from an underlying likelihood.
        :param observation: a new observation of time series.
        :return: an array of predictive posterior probabilities (densities).
        """
        assert self.__likelihood is not None, "Underlying likelihood must not be None"

        return self.__likelihood.predict(observation)

    def update(self, observation: np.float64) -> None:
        """
        Updates an underlying likelihood's state (calculates posterior parameters).
        :param observation: a new observation of time series.
        :return:
        """
        assert self.__likelihood is not None, "Underlying likelihood must not be None"

        self.__likelihood.update(observation)

    def clear(self) -> None:
        """
        Sets an underlying likelihood to None.
        :return:
        """
        self.__likelihood = None
