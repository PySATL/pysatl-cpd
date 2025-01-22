"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path
import logging
import yaml

from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from benchmarking.worker.common.statistics_calculation import StatisticsCalculation
from benchmarking.worker.common.threshold_calculation import ThresholdCalculation
from benchmarking.worker.worker import Worker
from benchmarking.algorithms.benchmarking_knn import BenchmarkingKNNAlgorithm
from benchmarking.scrubber.benchmarking_linear_scrubber import BenchmarkingLinearScrubber
from benchmarking.generator.generator import VerboseSafeDumper


class ThresholdCalculationWorker(Worker):
    def __init__(self, cpd_algorithm: BenchmarkingKNNAlgorithm, scrubber: BenchmarkingLinearScrubber, optimal_values_storage_path: Path, significance_level: float, sl_delta: float, sample_length: int, interval_length: int, logger: logging.Logger) -> None:
        self.__cpd_algorithm = cpd_algorithm
        self.__scrubber = scrubber
        self.__optimal_value_storage_path = optimal_values_storage_path
        self.__significance_level = significance_level
        self.__sl_delta = sl_delta
        self.__sample_length = sample_length
        self.__interval_length = interval_length

    def run(
        self,
        dataset_path: Path | None,
        results_path: Path,
    ) -> None:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        if dataset_path is not None:
            StatisticsCalculation.calculate_statistics(self.__cpd_algorithm, self.__scrubber, dataset_path, results_path)

        scrubber_metaparams = self.__scrubber.get_metaparameters()
        alg_metaparams = self.__cpd_algorithm.get_metaparameters()

        threshold = ThresholdCalculation.calculate_threshold(
            self.__significance_level,
            1.0,
            self.__sample_length,
            int(scrubber_metaparams["window_length"]),
            results_path,
            self.__interval_length,
            self.__sl_delta,
        )

        result_info = [{"config": {"algorithm": alg_metaparams, "scrubber": scrubber_metaparams}, "optimal_values": {"threshold": threshold}}]
        with open(self.__optimal_value_storage_path, "a") as outfile:
            yaml.dump(result_info, outfile, default_flow_style=False, sort_keys=False, Dumper=VerboseSafeDumper)
