"""
Module for implementation of CPD algorithm using KLIEP-based divergence estimation.
"""

__author__ = "Aleksandra Listkova"
__copyright__ = "Copyright (c) 2025 Aleksandra Listkova"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize  # type: ignore

from pysatl_cpd.core.algorithms.density.abstracts.density_based_algorithm import DensityBasedAlgorithm


class KliepAlgorithm(DensityBasedAlgorithm):
    def __init__(
        self,
        bandwidth: float = 1.0,
        regularization: float = 0.1,
        threshold: float = 1.1,
        max_iter: int = 100,
        min_window_size: int = 10,
    ) -> None:
        """
        Initializes a new instance of KLIEP-based change point detection algorithm.

        :param bandwidth: kernel bandwidth for density estimation.
        :param regularization: regularization coefficient for alpha optimization.
        :param threshold: threshold for change point detection.
        :param max_iter: maximum iterations for optimization solver.
        :param min_window_size: minimum segment size for reliable estimation.
        """
        super().__init__(min_window_size, threshold, bandwidth)
        self.regularization = regularization
        self.max_iter = max_iter

    def _compute_scores(self, window: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Computes KLIEP-based change point scores for each position in the window.

        :param window: input data window (1D array).
        :return: array of change point scores at each index.
        """
        n_points = window.shape[0]
        scores: npt.NDArray[np.float64] = np.zeros(n_points, dtype=np.float64)
        common_grid = self._build_common_grid(window)

        for i in range(self.min_window_size, n_points - self.min_window_size):
            before = window[:i]
            after = window[i:]

            before_density = self._kde_on_grid(before, self.bandwidth, common_grid)
            after_density = self._kde_on_grid(after, self.bandwidth, common_grid)

            alpha = self._optimize_alpha(after_density, before_density)
            scores[i] = np.mean(np.log(after_density + 1e-10)) - np.mean(np.log(before_density + 1e-10)) - alpha
        return scores

    def _optimize_alpha(
        self,
        test_density: npt.NDArray[np.float64],
        ref_density: npt.NDArray[np.float64]
    ) -> float:
        """
        Optimizes alpha parameter for density ratio estimation.

        :param test_density: KDE values for test segment (after potential CP).
        :param ref_density: KDE values for reference segment (before potential CP).
        :return: optimal alpha value for density ratio adjustment.
        """
        def loss(alpha_array: npt.NDArray[np.float64]) -> float:
            alpha = alpha_array[0]
            ratio = np.exp(np.log(test_density) - np.log(ref_density + 1e-10) - alpha)
            return float(-np.mean(np.log(ratio + 1e-10)) + self.regularization * alpha**2)

        initial_alpha = np.array([0.0], dtype=np.float64)
        bounds = [(0.0, None)]

        res = minimize(
            loss,
            x0=initial_alpha,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.max_iter}
        )
        return float(res.x[0])

    def _build_common_grid(self, window: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Creates evaluation grid for density estimation.

        :param window: input data window.
        :return: grid spanning data range with bandwidth-adjusted margins.
        """
        return np.linspace(
            np.min(window) - 3 * self.bandwidth,
            np.max(window) + 3 * self.bandwidth,
            1000,
            dtype=np.float64
        )

    def _kde_on_grid(
        self,
        observation: npt.NDArray[np.float64],
        bandwidth: float,
        grid: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Computes kernel density estimate on specified grid.

        :param observation: data points for KDE.
        :param bandwidth: kernel bandwidth parameter.
        :param grid: evaluation grid points.
        :return: density values at grid points.
        """
        n = observation.shape[0]
        diff = grid[:, np.newaxis] - observation
        kernel_vals = np.exp(-0.5 * (diff / bandwidth) ** 2)
        kde_vals = kernel_vals.sum(axis=1)
        return np.asarray(kde_vals / (n * bandwidth * np.sqrt(2 * np.pi)), dtype=np.float64)
