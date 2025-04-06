from collections.abc import Iterable
import numpy as np
from pysatl_cpd.core.algorithms.Entropies.abs import Algorithm


class ConditionalEntropyAlgorithm(Algorithm):
    """
    ConditionalEntropyAlgorithm detects change points in time series using conditional entropy.

    This algorithm evaluates the uncertainty of one time series (X) given another (Y) over sliding windows.
    Changes are detected when the conditional entropy shifts significantly.

    Attributes:
        conditional_data (np.ndarray): External reference time series Y used for conditioning.
        window_size (int): Size of each sliding window.
        step (int): Step size for sliding window.
        bins (int): Number of histogram bins for entropy calculation.
        threshold (float): Threshold for detecting changes in conditional entropy.
    """

    def __init__(
            self,
            conditional_data: np.ndarray,
            window_size: int = 40,
            step: int = 20,
            bins: int = 10,
            threshold: float = 0.3
    ):
        """
        Initializes the ConditionalEntropyAlgorithm.

        Args:
            conditional_data (np.ndarray): Conditioning time series Y.
            window_size (int): Size of sliding window for entropy calculation.
            step (int): Step size to shift the window.
            bins (int): Number of histogram bins for joint distribution estimation.
            threshold (float): Minimum change in entropy to consider a change point.
        """
        self._conditional_data = conditional_data
        self._window_size = window_size
        self._step = step
        self._bins = bins
        self._threshold = threshold
        self._change_points = []

    def detect(self, window: Iterable[float | np.float64]) -> int:
        """
        Detects the number of change points based on conditional entropy.

        Args:
            window (Iterable[float | np.float64]): Main time series X.

        Returns:
            int: Number of detected change points.

        Raises:
            ValueError: If conditional data is missing or mismatched in size.
        """
        if self._conditional_data is None:
            raise ValueError("Для вычисления условной энтропии необходимо предоставить условные данные.")

        time_series = np.array(list(window))
        if len(time_series) != len(self._conditional_data):
            raise ValueError("Размеры основных и условных данных должны совпадать.")

        entropy_values = self._sliding_conditional_entropy(time_series, self._conditional_data)
        changes = self._detect_change_points(entropy_values)
        return len(changes)

    def localize(self, window: Iterable[float | np.float64]) -> list[int]:
        """
        Localizes the positions of detected change points in the input time series.

        Args:
            window (Iterable[float | np.float64]): Main time series X.

        Returns:
            list[int]: Indices of localized change points.

        Raises:
            ValueError: If conditional data is missing or mismatched in size.
        """
        if self._conditional_data is None:
            raise ValueError("Для вычисления условной энтропии необходимо предоставить условные данные.")

        time_series = np.array(list(window))
        if len(time_series) != len(self._conditional_data):
            raise ValueError("Размеры основных и условных данных должны совпадать.")

        entropy_values = self._sliding_conditional_entropy(time_series, self._conditional_data)
        change_indices = self._detect_change_points(entropy_values)

        localized_points = []
        for idx in change_indices:
            actual_idx = idx * self._step + self._window_size // 2
            if actual_idx < len(time_series):
                localized_points.append(int(actual_idx))

        return localized_points

    def _sliding_conditional_entropy(self, time_series_x: np.ndarray, time_series_y: np.ndarray) -> np.ndarray:
        """
        Computes conditional entropy over sliding windows.

        Args:
            time_series_x (np.ndarray): Main signal X.
            time_series_y (np.ndarray): Conditioning signal Y.

        Returns:
            np.ndarray: Conditional entropy values for each window.
        """
        entropies = []
        for i in range(0, len(time_series_x) - self._window_size + 1, self._step):
            window_x = time_series_x[i:i + self._window_size]
            window_y = time_series_y[i:i + self._window_size]
            entropy_value = self._compute_conditional_entropy(window_x, window_y)
            entropies.append(entropy_value)
        return np.array(entropies)

    def _compute_conditional_entropy(self, time_series_x: np.ndarray, time_series_y: np.ndarray) -> float:
        """
        Estimates the conditional entropy H(X|Y) using histogram-based joint probabilities.

        Args:
            time_series_x (np.ndarray): Main signal segment.
            time_series_y (np.ndarray): Conditioning signal segment.

        Returns:
            float: Estimated conditional entropy H(X|Y).
        """
        hist2d, _, _ = np.histogram2d(time_series_x, time_series_y, bins=[self._bins, self._bins])
        joint_probability_matrix = hist2d / np.sum(hist2d)
        p_y = np.sum(joint_probability_matrix, axis=0)
        conditional_probability = np.divide(
            joint_probability_matrix,
            p_y,
            where=p_y != 0
        )
        H_X_given_Y = -np.nansum(joint_probability_matrix * np.log2(
            conditional_probability, where=conditional_probability > 0))
        return H_X_given_Y

    def _detect_change_points(self, entropy_values: np.ndarray) -> np.ndarray:
        """
        Detects significant changes in entropy values using the threshold.

        Args:
            entropy_values (np.ndarray): Conditional entropy values over time.

        Returns:
            np.ndarray: Indices of detected change points.
        """
        changes = np.where(np.abs(np.diff(entropy_values)) > self._threshold)[0]
        return changes