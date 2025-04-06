from collections.abc import Iterable
import numpy as np
from pysatl_cpd.core.algorithms.Entropies.abs import Algorithm


class ShannonEntropyAlgorithm(Algorithm):
    """
    ShannonEntropyAlgorithm detects change points in time series based on Shannon entropy.

    This method estimates the information content of signal segments using histogram-based
    Shannon entropy. A significant shift in entropy between windows indicates a potential change.

    Attributes:
        window_size (int): Size of the sliding window.
        step (int): Step size for moving the window.
        bins (int): Number of histogram bins used in entropy estimation.
        threshold (float): Threshold for detecting significant changes in entropy values.
    """

    def __init__(
        self,
        window_size: int = 40,
        step: int = 20,
        bins: int = 10,
        threshold: float = 0.3
    ):
        """
        Initializes the ShannonEntropyAlgorithm.

        Args:
            window_size (int): Length of each window to compute entropy.
            step (int): Step size for moving the window forward.
            bins (int): Number of bins for the histogram used in entropy calculation.
            threshold (float): Minimum change in entropy to consider a point of change.
        """
        self._window_size = window_size
        self._step = step
        self._bins = bins
        self._threshold = threshold
        self._change_points = []

    def detect(self, window: Iterable[float | np.float64]) -> int:
        """
        Detects the number of change points in a given time series.

        Args:
            window (Iterable[float | np.float64]): Input time series data.

        Returns:
            int: Number of detected change points.
        """
        time_series = np.array(list(window))
        entropy_values = self._sliding_entropy(time_series)
        changes = self._detect_change_points(entropy_values)
        return len(changes)

    def localize(self, window: Iterable[float | np.float64]) -> list[int]:
        """
        Returns the indices of localized change points in the time series.

        Args:
            window (Iterable[float | np.float64]): Input time series data.

        Returns:
            list[int]: List of time indices where change points are detected.
        """
        time_series = np.array(list(window))
        entropy_values = self._sliding_entropy(time_series)
        change_indices = self._detect_change_points(entropy_values)

        localized_points = []
        for idx in change_indices:
            actual_idx = idx * self._step + self._window_size // 2
            if actual_idx < len(time_series):
                localized_points.append(int(actual_idx))

        return localized_points

    def _sliding_entropy(self, time_series: np.ndarray) -> np.ndarray:
        """
        Computes Shannon entropy over sliding windows of the time series.

        Args:
            time_series (np.ndarray): Input signal as a NumPy array.

        Returns:
            np.ndarray: Array of entropy values computed over each window.
        """
        entropies = []
        for i in range(0, len(time_series) - self._window_size + 1, self._step):
            window = time_series[i:i + self._window_size]
            hist, _ = np.histogram(window, bins=self._bins, density=True)
            hist = hist / np.sum(hist)
            entropies.append(self._compute_entropy(hist))
        return np.array(entropies)

    def _compute_entropy(self, probabilities: np.ndarray) -> float:
        """
        Computes Shannon entropy for a given probability distribution.

        Args:
            probabilities (np.ndarray): Normalized histogram probabilities.

        Returns:
            float: Shannon entropy value.
        """
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def _detect_change_points(self, entropy_values: np.ndarray) -> np.ndarray:
        """
        Detects indices where entropy changes exceed the threshold.

        Args:
            entropy_values (np.ndarray): Sequence of computed entropy values.

        Returns:
            np.ndarray: Indices where significant entropy changes occur.
        """
        changes = np.where(np.abs(np.diff(entropy_values)) > self._threshold)[0]
        return changes