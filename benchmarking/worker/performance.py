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
from CPDShell.labeled_data import LabeledCPData
from CPDShell.shell import CPContainer, CPDProblem


class PerformanceWorker(Worker):
    def __init__(
        self,
        benchmark,
        cpd_algorithm: BenchmarkingKNNAlgorithm | BenchmarkingClassificationAlgorithm,
        scrubber: BenchmarkingLinearScrubber,
    ) -> None:
        self.__benchmark = benchmark
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
        assert dataset_path is not None

        data = list(LabeledCPData.read_generated_datasets(dataset_path).values())[0].raw_data
        shell = CPDProblem(data, cpd_algorithm=self.__cpd_algorithm, scrubber=self.__scrubber)
        self.__benchmark(shell.run_cpd)
