"""
Module for implementation of CPD algorithm using RULSIF-based divergence estimation.
"""

__author__ = "Aleksandra Listkova"
__copyright__ = "Copyright (c) 2025 Aleksandra Listkova"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve

from pysatl_cpd.core.algorithms.density.abstracts.density_based_algorithm import DensityBasedAlgorithm


class RulsifAlgorithm(DensityBasedAlgorithm):
    def __init__(
        self,
        alpha: float = 0.1,
        bandwidth: float = 1.0,
        lambda_reg: float = 0.1,
        threshold: float = 1.1,
        min_window_size: int = 10,
    ) -> None:
        """
        Initializes RULSIF-based change point detector.

        :param alpha: mixture coefficient (0-1) for reference/test densities
        :param bandwidth: kernel bandwidth for density estimation
        :param lambda_reg: L2 regularization strength
        :param threshold: detection sensitivity threshold
        :param min_window_size: minimum segment size requirement
        :raises ValueError: if alpha is not in (0,1)
        """
        super().__init__(min_window_size, threshold, bandwidth)
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha
        self.lambda_reg = lambda_reg

    def _compute_scores(self, window: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Computes RULSIF-based change point scores for each position.

        :param window: input data window (1D array)
        :return: array of divergence scores at each index
        """
        n_points = window.shape[0]
        scores: npt.NDArray[np.float64] = np.zeros(n_points, dtype=np.float64)
        for i in range(self.min_window_size, n_points - self.min_window_size):
            ref = window[:i]
            test = window[i:]
            K_ref = self._kernel_density_estimation(ref, self.bandwidth)
            K_test = self._kernel_density_estimation(test, self.bandwidth)
            H = (
                (1 - self.alpha) * (K_ref @ K_ref.T) / i
                + self.alpha * (K_test @ K_test.T) / (n_points - i)
                + self.lambda_reg * np.eye(K_ref.shape[0], dtype=np.float64)
            )
            h = K_test.mean(axis=1)
            theta = solve(H, h, assume_a='pos')
            density_ratio = theta @ K_test
            scores[i] = np.mean((density_ratio - 1) ** 2)
        return scores
