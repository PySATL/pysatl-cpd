from abc import abstractmethod
from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from pysatl_cpd.core.algorithms.abstract_algorithm import Algorithm

_TObjFunc: TypeAlias = Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float]


class DensityBasedAlgorithm(Algorithm):
    @staticmethod
    def _kernel_density_estimation(observation: npt.NDArray[np.float64], bandwidth: float) -> npt.NDArray[np.float64]:
        """Perform kernel density estimation on the given observations without fitting a model.

        :param observation: the data points for which to estimate the density.
        :param bandwidth: the bandwidth parameter for the kernel density estimation.

        :return: estimated density values for the observations.
        """
        n = len(observation)
        x_grid = np.linspace(np.min(observation) - 3 * bandwidth, np.max(observation) + 3 * bandwidth, 1000)
        kde_values = np.zeros_like(x_grid)
        for x in observation:
            kde_values += np.exp(-0.5 * ((x_grid - x) / bandwidth) ** 2)

        kde_values /= n * bandwidth * np.sqrt(2 * np.pi)
        return kde_values

    def _calculate_weights(
        self,
        test_value: npt.NDArray[np.float64],
        reference_value: npt.NDArray[np.float64],
        bandwidth: float,
        objective_function: _TObjFunc,
    ) -> npt.NDArray[np.float64]:
        """Calculate the weights based on the density ratio between test and reference values.

        :param test_value: the test data points.
        :param reference_value: the reference data points.
        :param bandwidth: the bandwidth parameter for the kernel density estimation.
        :param objective_function: the objective function to minimize.

        :return: the calculated density ratios normalized to their mean.
        """
        test_density = self._kernel_density_estimation(test_value, bandwidth)
        reference_density = self._kernel_density_estimation(reference_value, bandwidth)

        def objective_function_wrapper(alpha: npt.NDArray[np.float64], /) -> float:
            """Wrapper for the objective function to calculate the density ratio.

            :param alpha: relative parameter that controls the weighting between the numerator distribution
                and the denominator distribution in the density ratio estimation.

            :return: the value of the objective function to minimize.
            """
            objective_density_ratio = np.exp(test_density - reference_density - alpha)
            return objective_function(objective_density_ratio, alpha)

        res = minimize(objective_function_wrapper, np.zeros(len(test_value)), method="L-BFGS-B")
        optimized_alpha: npt.NDArray[np.float64] = res.x
        density_ratio: npt.NDArray[np.float64] = np.exp(test_density - reference_density - optimized_alpha)
        return density_ratio / np.mean(density_ratio)

    @abstractmethod
    def detect(self, window: npt.NDArray[np.float64]) -> int:
        # maybe rtype tuple[int]
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: list of right borders of window change points
        """
        raise NotImplementedError

    @abstractmethod
    def localize(self, window: npt.NDArray[np.float64]) -> list[int]:
        """Function for finding coordinates of change points in window

        :param window: part of global data for finding change points
        :return: list of window change points
        """
        raise NotImplementedError

    @staticmethod
    def evaluate_detection_accuracy(
        true_change_points: list[int], detected_change_points: list[int]
    ) -> dict[str, int | float]:
        """Evaluate the accuracy of change point detection.

        :param true_change_points: list of true change point indices.
        :param detected_change_points: list of detected change point indices.

        :return: a dictionary with evaluation metrics (precision, recall, F1 score).
        """
        true_positive = len(set(true_change_points) & set(detected_change_points))
        false_positive = len(set(detected_change_points) - set(true_change_points))
        false_negative = len(set(true_change_points) - set(detected_change_points))

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
