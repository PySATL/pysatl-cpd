from collections.abc import Iterable
from collections import Counter

import numpy as np

from pysatl_cpd.core.algorithms.Entropies.abs import Algorithm


class PermutationEntropyAlgorithm(Algorithm):
    """
    PermutationEntropyAlgorithm detects change points in time series using permutation entropy.

    This algorithm estimates the complexity of the time series in each sliding window and
    identifies significant changes by thresholding the difference in entropy values.

    Attributes:
        window_size (int): Size of each sliding window.
        step (int): Step size for sliding the window.
        embedding_dimension (int): Number of points in permutation vectors.
        time_delay (int): Delay between points in embedded vectors.
        threshold (float): Threshold for change detection in entropy differences.
    """

    def __init__(
            self,
            window_size: int = 40,
            step: int = 20,
            embedding_dimension: int = 3,
            time_delay: int = 1,
            threshold: float = 0.3
    ):
        """
        Initializes the PermutationEntropyAlgorithm.

        Args:
            window_size (int): Size of the window to compute entropy over.
            step (int): Step size to move the window.
            embedding_dimension (int): Length of the permutation vector.
            time_delay (int): Spacing between elements in the permutation vector.
            threshold (float): Threshold for detecting significant entropy shifts.
        """
        self._window_size = window_size
        self._step = step
        self._embedding_dimension = embedding_dimension
        self._time_delay = time_delay
        self._threshold = threshold
        self._change_points = []

    def detect(self, window: Iterable[float | np.float64]) -> int:
        """
        Detects the number of change points in a time series.

        Args:
            window (Iterable[float | np.float64]): Input time series.

        Returns:
            int: Number of detected change points.
        """
        time_series = np.array(list(window))
        entropy_values = self._sliding_permutation_entropy(time_series)
        changes = self._detect_change_points(entropy_values)
        return len(changes)

    def localize(self, window: Iterable[float | np.float64]) -> list[int]:
        """
        Returns the indices of localized change points.

        Args:
            window (Iterable[float | np.float64]): Input time series.

        Returns:
            list[int]: Indices of change points within the time series.
        """
        time_series = np.array(list(window))
        entropy_values = self._sliding_permutation_entropy(time_series)
        change_indices = self._detect_change_points(entropy_values)

        localized_points = []
        for idx in change_indices:
            actual_idx = idx * self._step + self._window_size // 2
            if actual_idx < len(time_series):
                localized_points.append(int(actual_idx))

        return localized_points

    def _sliding_permutation_entropy(self, time_series: np.ndarray) -> np.ndarray:
        """
        Calculates permutation entropy across sliding windows of a time series.

        Args:
            time_series (np.ndarray): Input signal array.

        Returns:
            np.ndarray: Array of entropy values for each window.
        """
        entropies = []
        for i in range(0, len(time_series) - self._window_size + 1, self._step):
            window = time_series[i:i + self._window_size]
            entropy_value = self._calculate_permutation_entropy(window)
            entropies.append(entropy_value)
        return np.array(entropies)

    def _calculate_permutation_entropy(self, time_series: np.ndarray) -> float:
        """
        Computes permutation entropy for a single window of the time series.

        Args:
            time_series (np.ndarray): Input data for one window.

        Returns:
            float: Estimated permutation entropy.
        """
        permutation_vectors = []
        for index in range(len(time_series) - self._embedding_dimension * self._time_delay):
            current_window = time_series[
                             index:index + self._embedding_dimension * self._time_delay:self._time_delay
                             ]
            permutation_vector = np.argsort(current_window)
            permutation_vectors.append(tuple(permutation_vector))

        permutation_counts = Counter(permutation_vectors)
        total_permutations = len(permutation_vectors)
        permutation_probabilities = [
            count / total_permutations for count in permutation_counts.values()
        ]
        permutation_entropy = -np.sum([
            probability * np.log2(probability)
            for probability in permutation_probabilities if probability > 0
        ])

        return permutation_entropy

    def _detect_change_points(self, entropy_values: np.ndarray) -> np.ndarray:
        """
        Detects change points based on significant differences in entropy values.

        Args:
            entropy_values (np.ndarray): Array of entropy values over time.

        Returns:
            np.ndarray: Indices of detected change points.
        """
        changes = np.where(np.abs(np.diff(entropy_values)) > self._threshold)[0]
        return changes