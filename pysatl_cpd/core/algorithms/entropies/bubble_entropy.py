from collections import Counter, deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class BubbleEntropyAlgorithm(OnlineAlgorithm):
    """
    **BubbleEntropyAlgorithm** detects change points in a time series using bubble entropy.

    The algorithm calculates bubble entropy values based on permutation entropy with varying embedding dimensions.
    It then detects significant changes based on predefined thresholds.

    :param window_size: Size of each sliding window.
    :param embedding_dimension: The embedding dimension used for calculating permutation entropy.
    :param time_delay: Time delay between elements in each state vector for calculating permutation entropy.
    :param threshold: Threshold for detecting changes based on entropy differences.

    **Attributes:**
    - `window_size` (int): Size of each sliding window.
    - `embedding_dimension` (int): The embedding dimension used for calculating permutation entropy.
    - `time_delay` (int): Time delay between elements in each state vector.
    - `threshold` (float): Threshold for change detection based on entropy shift.
    - `min_observations_for_detection` (int): Minimum number of observations required to detect a change point.
    - `_buffer` (deque): A buffer for storing the most recent observations.
    - `_entropy_values` (list): A list to store the calculated entropy values.
    - `_position` (int): The current position in the observation sequence.
    - `_last_change_point` (Optional[int]): The position of the last detected change point.
    """

    def __init__(
        self,
        window_size: int = 100,
        embedding_dimension: int = 3,
        time_delay: int = 1,
        threshold: float = 0.2,
    ):
        """
        Initializes the BubbleEntropyAlgorithm with the specified parameters.

        :param window_size: Size of each sliding window.
        :param embedding_dimension: The embedding dimension used for calculating permutation entropy.
        :param time_delay: Time delay between elements in each state vector for calculating permutation entropy.
        :param threshold: Threshold for detecting changes based on entropy differences.
        """
        self._window_size = window_size
        self._embedding_dimension = embedding_dimension
        self._time_delay = time_delay
        self._threshold = threshold

        self._buffer: deque[float] = deque(maxlen=window_size * 2)
        self._entropy_values: list[float] = []
        self._position: int = 0
        self._last_change_point: Optional[int] = None

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Processes the input observation to detect if a change point occurs in the time series.

        :param observation: A single observation or an array of observations.
        :return: `True` if a change point is detected, otherwise `False`.
        """
        if isinstance(observation, np.ndarray):
            for obs in observation:
                self._process_single_observation(float(obs))
        else:
            self._process_single_observation(float(observation))

        return self._last_change_point is not None

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Localizes the detected change point based on the observation.

        :param observation: A single observation or an array of observations.
        :return: The position of the detected change point, or `None` if no change point is detected.
        """
        change_detected = self.detect(observation)

        if change_detected:
            change_point = self._last_change_point
            self._last_change_point = None
            return change_point

        return None

    def _process_single_observation(self, observation: float) -> None:
        """
        Processes a single observation and updates the internal state. This method checks for significant deviations,
        computes bubble entropy, and detects change points when applicable.

        :param observation: The observation value to be processed.
        """
        threshold_value1 = 3.0
        threshold_value2 = 2.0

        if len(self._buffer) >= self._window_size // 2:
            buffer_mean = sum(list(self._buffer)[-self._window_size // 2 :]) / (self._window_size // 2)
            if abs(observation - buffer_mean) > threshold_value1:
                self._last_change_point = self._position

        self._buffer.append(observation)
        self._position += 1

        min_required = (self._embedding_dimension + 1) * self._time_delay + 1
        if len(self._buffer) < self._window_size or len(self._buffer) < min_required:
            return

        current_entropy = self._calculate_bubble_entropy(np.array(list(self._buffer)[-self._window_size :]))
        self._entropy_values.append(current_entropy)

        if len(self._entropy_values) >= threshold_value2:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])

            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

    def _calculate_bubble_entropy(self, time_series: npt.NDArray[np.float64]) -> float:
        """
        Calculates the bubble entropy of a time series by computing the difference in permutation entropy
        between two different embedding dimensions.

        :param time_series: The time series to analyze.
        :return: The computed bubble entropy value.
        """
        H_swaps_m = self._calculate_permutation_entropy(time_series, self._embedding_dimension)
        H_swaps_m_plus_1 = self._calculate_permutation_entropy(time_series, self._embedding_dimension + 1)

        denom = np.log((self._embedding_dimension + 1) / self._embedding_dimension)
        bubble_entropy = (H_swaps_m_plus_1 - H_swaps_m) / denom

        return float(bubble_entropy)

    def _calculate_permutation_entropy(self, time_series: npt.NDArray[np.float64], embedding_dimension: int) -> float:
        """
        Calculates the permutation entropy of a time series based on the given embedding dimension.

        :param time_series: The time series data to analyze.
        :param embedding_dimension: The embedding dimension for the state vectors.
        :return: The computed permutation entropy value.
        """
        permutation_vectors = []
        for index in range(len(time_series) - embedding_dimension * self._time_delay):
            current_window = time_series[index : index + embedding_dimension * self._time_delay : self._time_delay]
            permutation_vector = np.argsort(current_window)
            permutation_vectors.append(tuple(permutation_vector))

        permutation_counts = Counter(permutation_vectors)
        total_permutations = len(permutation_vectors)

        if total_permutations == 0:
            return float(0)

        permutation_probabilities = [count / total_permutations for count in permutation_counts.values()]
        permutation_entropy = -np.sum(
            [probability * np.log2(probability) for probability in permutation_probabilities if probability > 0]
        )

        return float(permutation_entropy)
