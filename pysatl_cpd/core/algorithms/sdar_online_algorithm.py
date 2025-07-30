from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm
from pysatl_cpd.core.algorithms.sdar.sdar import SDARmodel


class SDARAlgorithm(OnlineAlgorithm):
    """
    An online change point detection algorithm based on a two-level SDAR model.
    """

    def __init__(
        self, order: int = 2, smoothing_window_size: int = 3, forgetting_factor: float = 0.99, threshold: float = 5
    ) -> None:
        """
        Initializes the change point detection algorithm.
        :param order: the AR model order for both SDAR models.
        :param smoothing_window_size: the smoothing window size for both SDAR models.
        :param forgetting_factor: the forgetting factor (lambda) for both SDAR models.
        :param threshold: the threshold for the second-level anomaly score, above which a change point is detected.
        :return:
        """
        assert threshold >= 0, "Threshold must be non-negative."

        self.__threshold: np.float64 = np.float64(threshold)

        self.__first_model: SDARmodel = SDARmodel(order, forgetting_factor, smoothing_window_size)
        self.__second_model: SDARmodel = SDARmodel(order, forgetting_factor, smoothing_window_size)

        self.__first_scores: list[np.float64 | None] = []
        self.__second_scores: list[np.float64 | None] = []

        self.__current_time: int = 0
        self.__was_change_point: bool = False
        self.__change_point: Optional[int] = None

    def clear(self) -> None:
        self.__first_model.clear()
        self.__second_model.clear()

        self.__first_scores = []
        self.__second_scores = []

        self.__current_time = 0
        self.__was_change_point = False
        self.__change_point = None

    def __process_point(self, observation: npt.NDArray[np.float64]) -> None:
        """
        Processes a single observation, updating both models and checking for a change point.
        :param observation: new observation.
        :return:
        """
        self.__current_time += 1

        first_smooth_score = self.__first_model.get_smoothed_score(observation)
        self.__first_scores.append(first_smooth_score)
        if first_smooth_score is None:
            return

        score_as_input = np.array([first_smooth_score])
        second_smooth_score = self.__second_model.get_smoothed_score(score_as_input)
        self.__second_scores.append(second_smooth_score)
        if second_smooth_score is None:
            return

        if second_smooth_score >= self.__threshold:
            self.__handle_change_point()

    def __handle_change_point(self) -> None:
        """
        Handles the change point detection. It registers the location and resets the models' state.
        :return:
        """
        self.__was_change_point = True
        self.__change_point = self.__current_time

        self.__first_model.clear()
        self.__second_model.clear()

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Method for a step of detection of a change point.
        :param observation: new observation of a time series.
        :return: bool observation whether a change point was detected after processing the new observation.
        """
        if not isinstance(observation, np.ndarray):
            observation = np.array([observation])

        self.__process_point(observation)
        result = self.__was_change_point
        self.__was_change_point = False
        return result

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Method for a step of localization of a change point.
        :param observation: new observation of a time series
        :return: absolute location of a change point, acquired after processing the new observation,
        or None if there wasn't any.
        """
        if not isinstance(observation, np.ndarray):
            observation = np.array([observation])

        self.__process_point(observation)
        result = self.__change_point
        self.__was_change_point = False
        self.__change_point = None
        return result
