"""
Module for Bayesian online change point detection algorithm.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2025 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"

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
        self.detector = detector
        self.hazard = hazard
        self.likelihood = likelihood
        self.localizer = localizer
        self.window_size = learning_sample_size

        self.training_data: list[np.float64] = []
        self.data_history: list[np.float64] = []
        self.current_time = 0

        self.is_training: bool = True
        self.run_length_probs: np.ndarray = np.array([])

        self.__was_change_point = False
        self.__change_point: int | None = None

    def clear(self) -> None:
        """
        Clears the state of the algorithm's instance.
        :return:
        """
        self.training_data = []
        self.data_history = []
        self.current_time = 0

        self.is_training = True
        self.run_length_probs = np.array([])

        self.__was_change_point = False
        self.__change_point = None

    def __train(self, value: np.float64) -> None:
        self.training_data.append(value)
        if len(self.training_data) == self.window_size:
            self.likelihood.clear()
            self.detector.clear()

            self.likelihood.learn(self.training_data)
            self.is_training = False
            self.run_length_probs = np.array([1.0])

    def __bayesian_update(self, value: np.float64):
        predictive_prob = self.likelihood.predict(value)
        current_run_lengths = np.arange(len(self.run_length_probs))
        hazards = self.hazard.hazard(current_run_lengths)
        growth_probs = self.run_length_probs * (1 - hazards) * predictive_prob
        reset_prob = np.sum(self.run_length_probs * hazards * predictive_prob)
        new_probs = np.append(reset_prob, growth_probs)
        new_probs /= np.sum(new_probs)
        self.run_length_probs = new_probs
        self.likelihood.update(value)

    def __handle_localization(self) -> None:
        run_length = self.localizer.localize(self.run_length_probs)
        change_point_location = self.current_time - run_length
        self.training_data = self.data_history[-run_length:]
        self.data_history = self.data_history[-run_length:]
        self.__change_point = change_point_location

        self.likelihood.clear()
        self.detector.clear()
        self.is_training = True

        if len(self.training_data) >= self.window_size:
            self.training_data = self.training_data[: self.window_size]
            self.likelihood.learn(self.training_data)
            self.is_training = False
            self.run_length_probs = np.array([1.0])

            for value in self.data_history[self.window_size + 1 :]:
                self.__bayesian_update(value)

    def __handle_detection(self) -> None:
        self.data_history = self.data_history[-1:]
        self.training_data = self.data_history[:]
        self.likelihood.clear()
        self.detector.clear()
        self.is_training = True

    def __process_point(self, value: np.float64, with_localization: bool) -> None:
        """
        Universal method for processing of another value of a time series.
        :param value: new value of a time series.
        :param with_localization: whether the method was called for localization of a change point.
        :return:
        """
        self.data_history.append(value)
        self.current_time += 1

        if self.is_training:
            self.__train(value)
        else:
            self.__bayesian_update(value)
            detected = self.detector.detect(self.run_length_probs)

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

    def localize(self, value: np.float64 | npt.NDArray[np.float64]):
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
