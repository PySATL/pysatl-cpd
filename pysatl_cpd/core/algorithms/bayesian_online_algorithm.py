"""
Module for Bayesian online change point detection algorithm.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.bayesian.abstracts import IChangePointFunction, IHazard, ILikelihood, ILocalizer
from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm

__all__ = ["BayesianOnline"]


class BayesianOnline(OnlineAlgorithm[np.float64]):
    """
    Class for Bayesian online change point detection algorithm.
    """

    def __init__(
        self,
        hazard: IHazard,
        likelihood: ILikelihood,
        learning_sample_size: int,
        cp_function: IChangePointFunction,
        localizer: ILocalizer | None = None,
    ) -> None:
        self.__cpd_function = cp_function
        self.__hazard = hazard
        self.__likelihood = likelihood
        self.__localizer = localizer
        self.__learning_sample_size = learning_sample_size

        self.__training_data: list[np.float64] = []
        self.__data_history: list[np.float64] = []
        self.__current_time = 0

        self.__is_training: bool = True
        self.__run_length_probs: npt.NDArray[np.float64] = np.array([])

        self.__change_point: Optional[int] = None

    def process(self, observation: np.float64) -> float:
        """
        Universal method for processing of another observation of a time series.
        :param observation: new observation of a time series.
        :return: change point detection function value.
        """
        if observation is npt.NDArray[np.float64]:
            raise TypeError("Multivariate observations are not supported")

        self.__data_history.append(observation)
        self.__current_time += 1

        if self.__is_training:
            self.__learn(observation)
            return 0.0
        else:
            self.__bayesian_update(observation)
            func_value = self.__cpd_function.calculate(self.__run_length_probs)
            return func_value

    def reset(self):
        if self.__localizer is not None:
            self.__handle_localization()
        else:
            self.__handle_detection()

    def __clear_training_data(self) -> None:
        """
        Clears list of training data.
        :return:
        """
        self.__training_data = []

    def __learn(self, observation: np.float64) -> None:
        """
        Performs a learning step for a prediction model until the given learning sample size is achieved.
        :param observation: new observation of a time series.
        :return:
        """
        self.__training_data.append(observation)
        if len(self.__training_data) == self.__learning_sample_size:
            self.__likelihood.clear()
            self.__cpd_function.clear()

            self.__likelihood.learn(np.array(self.__training_data))
            self.__is_training = False
            self.__run_length_probs = np.array([1.0])

        assert len(self.__training_data) <= self.__learning_sample_size, (
            "Training data should not be longer than learning sample size"
        )

    def __bayesian_update(self, observation: np.float64) -> None:
        """
        Performs a bayesian update of the algorithm's state.
        :param observation: new observation of a time series.
        :return:
        """
        predictive_prob = self.__likelihood.predict(observation)
        hazards = self.__hazard.hazard(np.arange(self.__run_length_probs.shape[0], dtype=np.intp))

        growth_probs = self.__run_length_probs * (1 - hazards) * predictive_prob
        change_point_prob = np.sum(self.__run_length_probs * hazards * predictive_prob)
        new_probs = np.append(change_point_prob, growth_probs)

        evidence = np.sum(new_probs)
        if evidence == 0.0:
            self.__run_length_probs = np.zeros(self.__run_length_probs.shape[0])
            self.__run_length_probs[0] = 1.0
            return

        assert evidence > 0.0, "Evidence must be > 0.0"
        new_probs /= evidence
        assert np.all(np.logical_and(new_probs >= 0.0, new_probs <= 1.0))

        self.__run_length_probs = new_probs
        self.__likelihood.update(observation)

    def __handle_localization(self) -> None:
        """
        Handles localization of the change point. It includes acquiring location, updating stored data and state of the
        algorithm, training it if possible and building corresponding run length distribution.
        :return:
        """
        assert self.__localizer is not None, "Cannot handle localization if localizer is None"
        run_length = self.__localizer.localize(self.__run_length_probs)
        change_point_location = self.__current_time - run_length
        assert 0 <= change_point_location <= self.__current_time, (
            "Change point shouldn't be outside the available scope"
        )

        assert len(self.__data_history) >= run_length, "Run length shouldn't exceed available data length"
        self.__data_history = self.__data_history[-run_length:] if run_length > 0 else []
        self.__clear_training_data()
        self.__change_point = change_point_location

        self.__likelihood.clear()
        self.__cpd_function.clear()
        self.__is_training = True

        observations_to_learn = min(len(self.__data_history), self.__learning_sample_size)
        data_to_train = self.__data_history[:observations_to_learn]

        # Learning as much as we can until we reach learning sample size limit
        for observation in data_to_train:
            self.__learn(observation)

        # Modeling run length probabilities on the rest data
        if len(self.__data_history) >= self.__learning_sample_size:
            self.__is_training = False
            for observation in self.__data_history[self.__learning_sample_size :]:
                self.__bayesian_update(observation)

    def __handle_detection(self) -> None:
        """
        Handles detection of the change point. It includes updating stored data and state of the algorithm.
        :return:
        """
        self.__data_history = self.__data_history[-1:]
        self.__clear_training_data()
        self.__likelihood.clear()
        self.__cpd_function.clear()
        self.__is_training = True
        self.__learn(self.__data_history[-1])
