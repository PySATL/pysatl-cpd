"""
Module for implementation of CPD algorithm using KLIEP-based divergence estimation.
"""

__author__ = "Aleksandra Listkova"
__copyright__ = "Copyright (c) 2025 Aleksandra Listkova"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from typing import List, Any
from typing import cast

from pysatl_cpd.core.algorithms.abstract_algorithm import Algorithm
from pysatl_cpd.core.algorithms.density.abstracts.density_based_algorithm import DensityBasedAlgorithm


class KliepAlgorithm(Algorithm):
    def __init__(
        self,
        bandwidth: float = 1.0,
        regularization: float = 0.1,
        threshold: float = 1.1,
        max_iter: int = 100,
        min_window_size: int = 10
    ) -> None:
        """
        Initializes a new instance of KLIEP based change point detection algorithm.

        :param bandwidth: the bandwidth parameter for the kernel density estimation.
        :param regularization: L2 regularization coefficient for the KLIEP optimization.
        :param threshold: detection threshold for significant change points.
        :param max_iter: maximum number of iterations for the L-BFGS-B optimizer.
        :param min_window_size: minimum size of data segments to consider.
        """
        self.bandwidth = bandwidth
        self.regularisation = regularization
        self.threshold = threshold
        self.max_iter = max_iter
        self.min_window_size = min_window_size

    def detect(self, window: npt.NDArray[np.float64]) -> int:
        """
        Finds change points in the given window.

        :param window: input data window for change point detection.
        :return: number of detected change points in the window.
        """
        return len(self.localize(window))

    def localize(self, window: npt.NDArray[np.float64]) -> list[int]:
        """
        Identifies and returns the locations of change points in the window.

        :param window: input data window for change point localization.
        :return: list of indices where change points were detected.
        """
        window = self._validate_window(window)
        if len(window) < self.min_window_size:
            return []

        scores = self._compute_kliep_scores(window)
        return self._find_change_points(scores)

    def _validate_window(self, window: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        """
        Validates and prepares the input window for processing.

        :param window: input data window.
        :return: validated window in 2D format.
        """
        window = np.asarray(window, dtype=np.float64)
        if window.ndim == 1:
            window = window.reshape(-1, 1)
        return window

    def _compute_kliep_scores(self, window: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Computes KLIEP anomaly scores for each point in the window.

        :param window: validated input data window.
        :return: array of KLIEP scores for each point.
        """
        n_points = window.shape[0]
        scores = np.zeros(n_points, dtype=np.float64)

        for i in range(self.min_window_size, n_points - self.min_window_size):
            before = window[:i]
            after = window[i:]

            before_density = DensityBasedAlgorithm._kernel_density_estimation(
                before, self.bandwidth
            )
            after_density = DensityBasedAlgorithm._kernel_density_estimation(
                after, self.bandwidth
            )

            alpha = self._optimize_alpha(after_density, before_density)
            scores[i] = np.mean(np.exp(after_density - before_density - alpha))

        return scores

    def _optimize_alpha(
            self,
            test_density: npt.NDArray[np.float64],
            ref_density: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Optimizes the alpha parameters for KLIEP density ratio estimation.

        :param test_density: density estimates for the test window.
        :param ref_density: density estimates for the reference window.
        :return: optimized alpha parameters.
        """
        def loss(alpha: npt.NDArray[np.float64]) -> float:
            """Objective function for KLIEP optimization."""
            ratio = np.exp(test_density - ref_density - alpha)
            loss_val = -np.mean(np.log(ratio)) + self.regularisation * np.sum(alpha**2)
            return float(loss_val)

        initial_alpha = np.zeros_like(test_density, dtype=np.float64)
        bounds = [(0, None)] * len(test_density)

        res = minimize(
            loss,
            initial_alpha,
            method='L-BFGS-B',
            options={'maxiter': self.max_iter},
            bounds=bounds
        )
        return cast(npt.NDArray[np.float64], res.x)

    def _find_change_points(self, scores: npt.NDArray[np.float64]) -> List[int]:
        """
        Identifies change points from computed KLIEP scores.

        :param scores: array of KLIEP scores for each point.
        :return: list of detected change point indices.
        """
        candidates = np.where(scores > self.threshold)[0]
        if len(candidates) == 0:
            return []

        change_points = [candidates[0]]
        for points in candidates[1:]:
            if points - change_points[-1] > self.min_window_size:
                change_points.append(points)

        return change_points
