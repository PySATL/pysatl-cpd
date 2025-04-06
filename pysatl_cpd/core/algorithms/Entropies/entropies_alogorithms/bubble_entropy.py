from collections.abc import Iterable
from collections import Counter

import numpy as np

from pysatl_cpd.core.algorithms.Entropies.abs import Algorithm


class BubbleEntropyAlgorithm(Algorithm):
    """
    BubbleEntropyAlgorithm implements a change point detection algorithm based on Bubble Entropy.

    This method calculates entropy differences over sliding windows and identifies points of
    structural change in time series using a threshold on the entropy gradient.

    Attributes:
        window_size (int): Size of the sliding window.
        step (int): Step size for the sliding window.
        embedding_dimension (int): Embedding dimension for permutation entropy.
        time_delay (int): Time delay between elements in embedded vectors.
        threshold (float): Threshold for detecting significant entropy changes.
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
        Initializes the BubbleEntropyAlgorithm with configuration parameters.

        Args:
            window_size (int): Length of each sliding window.
            step (int): Step between successive windows.
            embedding_dimension (int): Dimensionality for permutation vectors.
            time_delay (int): Lag between elements in permutation vectors.
            threshold (float): Minimum change in entropy to consider a change point.
        """
        self._window_size = window_size
        self._step = step
        self._embedding_dimension = embedding_dimension
        self._time_delay = time_delay
        self._threshold = threshold
        self._change_points = []

    def detect(self, window: Iterable[float | np.float64]) -> int:
        """
        Detects the number of change points in the given time series window.

        Args:
            window (Iterable[float | np.float64]): Input time series data.

        Returns:
            int: Number of detected change points.
        """
        time_series = np.array(list(window))
        entropy_values = self._sliding_bubble_entropy(time_series)
        changes = self._detect_change_points(entropy_values)
        return len(changes)

    def localize(self, window: Iterable[float | np.float64]) -> list[int]:
        """
        Returns the indices of localized change points in the input time series.

        Args:
            window (Iterable[float | np.float64]): Input time series data.

        Returns:
            list[int]: Indices of detected change points in the original time series.
        """
        time_series = np.array(list(window))
        entropy_values = self._sliding_bubble_entropy(time_series)
        change_indices = self._detect_change_points(entropy_values)

        localized_points = []
        for idx in change_indices:
            actual_idx = idx * self._step + self._window_size // 2
            if actual_idx < len(time_series):
                localized_points.append(int(actual_idx))

        return localized_points

    def _sliding_bubble_entropy(self, time_series: np.ndarray) -> np.ndarray:
        """
        Calculates bubble entropy over sliding windows of the time series.

        Args:
            time_series (np.ndarray): Input array of time series data.

        Returns:
            np.ndarray: Array of entropy values corresponding to each window.
        """
        entropies = []
        for i in range(0, len(time_series) - self._window_size + 1, self._step):
            window = time_series[i:i + self._window_size]
            entropy_value = self._calculate_bubble_entropy(window)
            entropies.append(entropy_value)
        return np.array(entropies)

    def _calculate_bubble_entropy(self, time_series: np.ndarray) -> float:
        """
        Computes the bubble entropy for a given time series window.

        Args:
            time_series (np.ndarray): Input segment of time series.

        Returns:
            float: Calculated bubble entropy value.
        """
        H_swaps_m = self._calculate_permutation_entropy(time_series, self._embedding_dimension)
        H_swaps_m_plus_1 = self._calculate_permutation_entropy(time_series, self._embedding_dimension + 1)

        bubble_entropy = (H_swaps_m_plus_1 - H_swaps_m) / np.log(
            (self._embedding_dimension + 1) / self._embedding_dimension)
        return bubble_entropy

    def _calculate_permutation_entropy(self, time_series: np.ndarray, embedding_dimension: int) -> float:
        """
        Computes the permutation entropy of a time series for a given embedding dimension.

        Args:
            time_series (np.ndarray): Time series window.
            embedding_dimension (int): Dimension of embedded permutation vectors.

        Returns:
            float: Permutation entropy value.
        """
        permutation_vectors = []
        for index in range(len(time_series) - embedding_dimension * self._time_delay):
            current_window = time_series[index:index + embedding_dimension * self._time_delay:self._time_delay]
            permutation_vector = np.argsort(current_window)
            permutation_vectors.append(tuple(permutation_vector))

        permutation_counts = Counter(permutation_vectors)
        total_permutations = len(permutation_vectors)
        permutation_probabilities = [count / total_permutations for count in permutation_counts.values()]
        permutation_entropy = -np.sum([
            probability * np.log2(probability)
            for probability in permutation_probabilities if probability > 0
        ])

        return permutation_entropy

    def _detect_change_points(self, entropy_values: np.ndarray) -> np.ndarray:
        """
        Detects significant changes in entropy values based on threshold.

        Args:
            entropy_values (np.ndarray): Series of entropy values.

        Returns:
            np.ndarray: Indices where significant changes are detected.
        """
        changes = np.where(np.abs(np.diff(entropy_values)) > self._threshold)[0]
        return changes