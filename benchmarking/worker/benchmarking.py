"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path

from benchmarking.algorithms.benchmarking_knn import BenchmarkingKNNAlgorithm
from benchmarking.algorithms.benchmarking_classification import BenchmarkingClassificationAlgorithm
from benchmarking.scrubber.benchmarking_linear_scrubber import BenchmarkingLinearScrubber
from benchmarking.worker.common.statistics_calculation import StatisticsCalculation
from benchmarking.worker.worker import Worker


class BenchmarkingWorker(Worker):
    def __init__(
        self,
        cpd_algorithm: BenchmarkingKNNAlgorithm | BenchmarkingClassificationAlgorithm,
        scrubber: BenchmarkingLinearScrubber,
        expected_change_points: list[int],
    ) -> None:
        self.__expected_change_points = expected_change_points
        self.__scrubber = scrubber
        self.__cpd_algorithm = cpd_algorithm

    def run(
        self,
        dataset_path: Path | None,
        results_path: Path,
    ) -> None:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        # TODO: If dataset_path is not None ...
        assert dataset_path is not None, "Dataset path should not be None"

        # TODO: Statistics calculation saves all metrics. Should it be the responsibility of worker?
        StatisticsCalculation.calculate_statistics(self.__cpd_algorithm, self.__scrubber, dataset_path, results_path)
