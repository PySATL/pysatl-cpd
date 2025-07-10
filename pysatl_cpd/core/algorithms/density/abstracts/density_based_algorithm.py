from abc import abstractmethod
from typing import Any, Callable, TypeAlias

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.abstract_algorithm import Algorithm

_TObjFunc: TypeAlias = Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]
_TMetrics: TypeAlias = dict[str, int | float]

class DensityBasedAlgorithm(Algorithm):
    """Abstract base class for density-based change point detection algorithms.

    Provides common infrastructure for methods that detect change points by
    analyzing probability density changes in data segments.
    """
    def __init__(
        self,
        min_window_size: int = 10,
        threshold: float = 1.1,
        bandwidth: float = 1.0
    ) -> None:
        """
        Initializes density-based change point detector.

        :param min_window_size: minimum data points required in each segment
        :param threshold: detection sensitivity (higher = fewer detections)
        :param bandwidth:kernel bandwidth for density estimation
        """
        self.min_window_size = min_window_size
        self.threshold = threshold
        self.bandwidth = bandwidth

    def detect(self, window: npt.NDArray[np.float64]) -> int:
        """Counts change points in the given data window.

        :param window: input data array (1D or 2D)
        :return: number of detected change points
        """
        return len(self.localize(window))

    def localize(self, window: npt.NDArray[np.float64]) -> list[int]:
        """Identifies positions of change points in the data window.

        :param window: input data array (1D or 2D)
        :return: list of change point indices
        """
        window = self._validate_window(window)
        if not self._is_window_valid(window):
            return []
        scores = self._compute_scores(window)
        return self._find_change_points(scores)

    def _validate_window(self, window: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        """Ensures input window meets processing requirements.

        :param window: raw input data
        :return: validated 2D float64 array
        """
        window_arr = np.asarray(window, dtype=np.float64)
        if window_arr.ndim == 1:
            window_arr = window_arr.reshape(-1, 1).astype(np.float64)
        return np.array(window_arr, dtype=np.float64)

    def _find_change_points(self, scores: npt.NDArray[np.float64]) -> list[int]:
        """Filters candidate points using threshold and minimum separation.

        :param scores: change point scores for each position
        :return: filtered list of change point indices
        """
        candidates = np.where(scores > self.threshold)[0]
        if not candidates.size:
            return []
        change_points = [int(candidates[0])]
        for point in candidates[1:]:
            if point - change_points[-1] > self.min_window_size:
                change_points.append(int(point))
        return change_points

    def _is_window_valid(self, window: npt.NDArray[np.float64]) -> bool:
        """Verifies window meets minimum size requirements.

        :param window: input data window
        :return: True if window can be processed, else False
        """
        return len(window) >= 2 * self.min_window_size

    @staticmethod
    def _kernel_density_estimation(
        observation: npt.NDArray[np.float64],
        bandwidth: float
    ) -> npt.NDArray[np.float64]:
        """Computes kernel density estimate using Gaussian kernels.

        :param observation: data points for density estimation
        :param bandwidth: smoothing parameter for KDE
        :return: density values at evaluation points
        """
        n = observation.shape[0]
        x_grid: npt.NDArray[np.float64] = np.linspace(
            np.min(observation) - 3*bandwidth,
            np.max(observation) + 3*bandwidth,
            1000,
            dtype=np.float64
        )

        diff = x_grid[:, np.newaxis] - observation
        kernel_vals = np.exp(-0.5 * (diff / bandwidth) ** 2)
        kde_vals = kernel_vals.sum(axis=1)

        return np.asarray(kde_vals / (n * bandwidth * np.sqrt(2*np.pi)), dtype=np.float64)

    @abstractmethod
    def _compute_scores(self, window: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Computes change point scores.

        :param window: validated input data
        :return: array of change point scores
        """
        raise NotImplementedError

    @staticmethod
    def evaluate_detection_accuracy(
        true_change_points: list[int],
        detected_change_points: list[int]
    ) -> _TMetrics:
        """Computes detection performance metrics.

        :param true_change_points: ground truth change points
        :param detected_change_points: algorithm-detected change points
        :return: dictionary containing precision, recall, F1, and error counts
        """
        true_positives = len(set(true_change_points) & set(detected_change_points))
        false_positives = len(set(detected_change_points) - set(true_change_points))
        false_negatives = len(set(true_change_points) - set(detected_change_points))

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0 else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0 else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positive": true_positives,
            "false_positive": false_positives,
            "false_negative": false_negatives,
        }
