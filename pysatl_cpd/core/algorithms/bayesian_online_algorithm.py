"""
Module for Bayesian online change point detection algorithm.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.bayesian.abstracts.idetector import IDetector
from pysatl_cpd.core.algorithms.bayesian.abstracts.ihazard import IHazard
from pysatl_cpd.core.algorithms.bayesian.abstracts.ilikelihood import ILikelihood
from pysatl_cpd.core.algorithms.bayesian.abstracts.ilocalizer import ILocalizer
from pysatl_cpd.core.algorithms.online_algorithm import OnlineCpdAlgorithm


class BayesianOnlineCpd(OnlineCpdAlgorithm):
    """
    Class for Bayesian online change point detection algorithm.
    """

    def __init__(
        self,
        hazard: IHazard,
        likelihood: ILikelihood,
        learning_sample_size: int,
        detector: IDetector,
        localizer: ILocalizer,
    ) -> None:
        self.__detector = detector
        self.__hazard = hazard
        self.__likelihood = likelihood
        self.__localizer = localizer
        self.__learning_sample_size = learning_sample_size

        self.__training_data: list[np.float64] = []
        self.__data_history: list[np.float64] = []
        self.__current_time = 0

        self.__is_training: bool = True
        self.__run_length_probs: npt.NDArray[np.float64] = np.array([])

        self.__was_change_point = False
        self.__change_point: int | None = None

    def clear(self) -> None:
        """
        Clears the state of the algorithm's instance.
        :return:
        """
        self.__training_data = []
        self.__data_history = []
        self.__current_time = 0

        self.__is_training = True
        self.__run_length_probs = np.array([])

        self.__was_change_point = False
        self.__change_point = None

    def __learn(self, value: np.float64) -> None:
        """
        Performs a learning step for a prediction model until the given learning sample size is achieved.
        :param value: new value of a time series.
        :return:
        """
        self.__training_data.append(value)
        if len(self.__training_data) == self.__learning_sample_size:
            self.__likelihood.clear()
            self.__detector.clear()

            self.__likelihood.learn(np.array(self.__training_data))
            self.__is_training = False
            self.__run_length_probs = np.array([1.0])

    def __bayesian_update(self, value: np.float64) -> None:
        """
        Performs a bayesian update of the algorithm's state.
        :param value: new value of a time series.
        :return:
        """
        predictive_prob = self.__likelihood.predict(value)
        hazards = self.__hazard.hazard(np.arange(self.__run_length_probs.shape[0], dtype=np.intp))
        growth_probs = self.__run_length_probs * (1 - hazards) * predictive_prob
        reset_prob = np.sum(self.__run_length_probs * hazards * predictive_prob)
        new_probs = np.append(reset_prob, growth_probs)
        new_probs /= np.sum(new_probs)
        self.__run_length_probs = new_probs
        self.__likelihood.update(value)

    def __handle_localization(self) -> None:
        """
        Handles localization of the change point. It includes acquiring location, updating stored data and state of the
        algorithm, training it if possible and building corresponding run length distribution.
        :return:
        """
        run_length = self.__localizer.localize(self.__run_length_probs)
        change_point_location = self.__current_time - run_length
        self.__training_data = self.__data_history[-run_length:]
        self.__data_history = self.__data_history[-run_length:]
        self.__change_point = change_point_location

        self.__likelihood.clear()
        self.__detector.clear()
        self.__is_training = True

        if len(self.__training_data) >= self.__learning_sample_size:
            self.__training_data = self.__training_data[: self.__learning_sample_size]
            for value in self.__training_data:
                self.__learn(value)

            for value in self.__data_history[self.__learning_sample_size :]:
                self.__bayesian_update(value)

    def __handle_detection(self) -> None:
        """
        Handles detection of the change point. It includes updating stored data and state of the algorithm.
        :return:
        """
        self.__data_history = self.__data_history[-1:]
        self.__training_data = self.__data_history[:]
        self.__likelihood.clear()
        self.__detector.clear()
        self.__is_training = True
        self.__learn(self.__training_data[-1])

    def __process_point(self, value: np.float64, with_localization: bool) -> None:
        """
        Universal method for processing of another value of a time series.
        :param value: new value of a time series.
        :param with_localization: whether the method was called for localization of a change point.
        :return:
        """
        self.__data_history.append(value)
        self.__current_time += 1

        if self.__is_training:
            self.__learn(value)
        else:
            self.__bayesian_update(value)
            detected = self.__detector.detect(self.__run_length_probs)

            if not detected:
                return

            self.__was_change_point = True
            if with_localization:
                self.__handle_localization()
            else:
                self.__handle_detection()

    def detect(self, value: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Performs a change point detection after processing another value of a time series.
        :param value: new value of a time series. Note: multivariate time series aren't supported for now.
        :return: whether a change point was detected after processing the new value.
        """
        if value is npt.NDArray[np.float64]:
            raise TypeError("Multivariate values are not supported")

        self.__process_point(np.float64(value), False)
        result = self.__was_change_point
        self.__was_change_point = False
        return result

    def localize(self, value: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Performs a change point localization after processing another value of a time series.
        :param value: new value of a time series.
        :return: location of a change point, acquired after processing the new value, or None if there wasn't any.
        """
        if value is npt.NDArray[np.float64]:
            raise TypeError("Multivariate values are not supported")

        self.__process_point(np.float64(value), True)
        result = self.__change_point
        self.__was_change_point = False
        self.__change_point = None
        return result
