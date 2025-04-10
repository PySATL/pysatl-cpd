from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class ConditionalEntropyAlgorithm(OnlineAlgorithm):
    """
    **ConditionalEntropyAlgorithm** detects change points in a time series using conditional entropy.

    This algorithm calculates entropy values based on joint and
    conditional probability distributions of paired observations
    and detects significant changes based on predefined thresholds.

    :param window_size: Size of each sliding window.
    :param bins: Number of bins used to create histograms for the joint probability distribution.
    :param threshold: Threshold for detecting changes based on entropy differences.

    **Attributes:**
    - `window_size` (int): Size of each sliding window.
    - `bins` (int): Number of histogram bins used for entropy calculation.
    - `threshold` (float): Threshold for change detection based on entropy shift.
    - `_buffer_x` (deque): A buffer for storing the X observations.
    - `_buffer_y` (deque): A buffer for storing the Y observations.
    - `_entropy_values` (list): A list that stores the calculated entropy values.
    - `_position` (int): The current position (or index) in the observation sequence.
    - `_last_change_point` (Optional[int]): The last position where a change point was detected.
    - `_prev_x` (Optional[float]): The previous X observation value.
    - `_prev_y` (Optional[float]): The previous Y observation value.
    - `_constant_value_count` (int): A counter for consecutive constant values.

    **Methods:**

    **`detect(window: Iterable[float | np.float64]) -> int`**
    Detects the number of change points in a time series.

    :param window: Input time series.
    :return: Number of detected change points.
    """

    def __init__(
        self,
        window_size: int = 40,
        bins: int = 10,
        threshold: float = 0.3,
    ):
        """
        Initializes the ConditionalEntropyAlgorithm with the specified parameters.

        :param window_size: Size of each sliding window.
        :param bins: Number of bins used to create histograms for the joint probability distribution.
        :param threshold: Threshold for detecting changes based on entropy differences.
        """
        self._window_size = window_size
        self._bins = bins
        self._threshold = threshold

        self._buffer_x: deque[float] = deque(maxlen=window_size * 2)
        self._buffer_y: deque[float] = deque(maxlen=window_size * 2)
        self._entropy_values: list[float] = []
        self._position: int = 0
        self._last_change_point: Optional[int] = None
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None
        self._constant_value_count: int = 0

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Processes a single observation or an array of observations to detect if a change point occurs in the time series

        :param observation: A single observation or an array of observations to be processed.
        :return: `True` if a change point is detected, otherwise `False`.
        """
        min_param = 2
        if isinstance(observation, np.ndarray) and observation.ndim > 0:
            if observation.shape[0] < min_param:
                raise ValueError("Observation array must contain both X and Y values")
            x_obs = float(observation[0])
            y_obs = float(observation[1])
            self._process_observation_pair(x_obs, y_obs)
        else:
            raise ValueError("Observation must be an array containing both X and Y values")

        return self._last_change_point is not None

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Localizes the detected change point based on the observation provided.

        :param observation: A single observation or an array of observations.
        :return: The position of the detected change point, or `None` if no change point is detected.
        """
        change_detected = self.detect(observation)

        if change_detected:
            change_point = self._last_change_point
            self._last_change_point = None
            return change_point

        return None

    def _process_observation_pair(self, x_obs: float, y_obs: float) -> None:
        """
        Processes a pair of observations (x and y) and updates the internal state.
        This method checks for constant values
        significant deviations from the moving averages, and computes conditional
        entropy for the sliding window to detect changes

        :param x_obs: The X observation value to be processed
        :param y_obs: The Y observation value to be processed
        """
        constant_value_threshold = 10
        significant_deviation_threshold = 3
        max_entropy_value = 2
        _epsilon_ = 1e-10

        self._handle_constant_values(x_obs, y_obs, constant_value_threshold)
        self._process_buffer_deviations(x_obs, y_obs, significant_deviation_threshold)
        self._check_correlation_change(x_obs, y_obs, _epsilon_)

        self._buffer_x.append(x_obs)
        self._buffer_y.append(y_obs)
        self._position += 1

        if len(self._buffer_x) < self._window_size:
            return

        self._compute_entropy(max_entropy_value)

    def _handle_constant_values(self, x_obs: float, y_obs: float, constant_value_threshold: int) -> None:
        """
        Checks if the current observations are constant (i.e., the same as the previous ones).
        If so, it increments the counter
        for constant values and detects a change point when the count exceeds the threshold.

        :param x_obs: The current X observation.
        :param y_obs: The current Y observation.
        :param constant_value_threshold: The threshold for detecting constant values.
        """
        if self._prev_x is not None and self._prev_y is not None:
            if x_obs == self._prev_x and y_obs == self._prev_y:
                self._constant_value_count += 1
            else:
                if self._constant_value_count >= constant_value_threshold:
                    self._last_change_point = self._position
                self._constant_value_count = 0

        self._prev_x = x_obs
        self._prev_y = y_obs

    def _process_buffer_deviations(self, x_obs: float, y_obs: float, significant_deviation_threshold: int) -> None:
        """
        Checks if the current observations deviate significantly from
        the moving averages of the previous values in the buffer.
        If a significant deviation is detected, a change point is recorded.

        :param x_obs: The current X observation.
        :param y_obs: The current Y observation.
        :param significant_deviation_threshold: The threshold for detecting significant deviations.
        """
        if len(self._buffer_x) >= self._window_size // 2:
            buffer_x_mean = sum(list(self._buffer_x)[-self._window_size // 2 :]) / (self._window_size // 2)
            buffer_y_mean = sum(list(self._buffer_y)[-self._window_size // 2 :]) / (self._window_size // 2)

            if (
                abs(x_obs - buffer_x_mean) > significant_deviation_threshold
                or abs(y_obs - buffer_y_mean) > significant_deviation_threshold
            ):
                self._last_change_point = self._position

    def _check_correlation_change(self, x_obs: float, y_obs: float, _epsilon_: float) -> None:
        """
        Checks for changes in the correlation between the most recent observations in the buffer.
         If a significant change in
        correlation is detected, a change point is recorded.

        :param x_obs: The current X observation.
        :param y_obs: The current Y observation.
        :param _epsilon_: A small value to avoid division by zero when calculating standard deviations.
        """
        threshold = 0.3

        if len(self._buffer_x) >= self._window_size // 4:
            recent_x = list(self._buffer_x)[-self._window_size // 4 :]
            recent_y = list(self._buffer_y)[-self._window_size // 4 :]

            std_x = np.std(recent_x)
            std_y = np.std(recent_y)

            if std_x > _epsilon_ and std_y > _epsilon_:
                try:
                    corr_before = np.corrcoef(recent_x, recent_y)[0, 1]

                    test_x = [*recent_x, x_obs]
                    test_y = [*recent_y, y_obs]

                    corr_after = np.corrcoef(test_x, test_y)[0, 1]

                    if (
                        not np.isnan(corr_before)
                        and not np.isnan(corr_after)
                        and abs(corr_after - corr_before) > threshold
                    ):
                        self._last_change_point = self._position
                except Exception:
                    pass

    def _compute_entropy(self, max_entropy_value: int) -> None:
        """
        Computes the conditional entropy for the most recent observations in the buffer.
        If the number of entropy values exceeds
        a predefined threshold, it checks if the difference between the latest
        two entropy values is significant, indicating a change point.

        :param max_entropy_value: The maximum number of entropy values to consider when detecting changes.
        """
        window_x = np.array(list(self._buffer_x)[-self._window_size :])
        window_y = np.array(list(self._buffer_y)[-self._window_size :])

        current_entropy = self._compute_conditional_entropy(window_x, window_y)
        self._entropy_values.append(current_entropy)

        if len(self._entropy_values) >= max_entropy_value:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])

            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

    def _compute_conditional_entropy(
        self, time_series_x: npt.NDArray[np.float64], time_series_y: npt.NDArray[np.float64]
    ) -> float:
        """
        Computes the conditional entropy of the time series using joint and conditional probability distributions.

        :param time_series_x: The X time series to be analyzed.
        :param time_series_y: The Y time series to be analyzed.
        :return: The calculated conditional entropy value.
        """
        hist2d, _, _ = np.histogram2d(time_series_x, time_series_y, bins=[self._bins, self._bins])
        joint_probability_matrix = hist2d / np.sum(hist2d)
        p_y = np.sum(joint_probability_matrix, axis=0)
        conditional_probability = np.divide(joint_probability_matrix, p_y, where=p_y != 0)
        H_X_given_Y = -np.nansum(
            joint_probability_matrix * np.log2(conditional_probability, where=conditional_probability > 0)
        )
        return float(H_X_given_Y)
