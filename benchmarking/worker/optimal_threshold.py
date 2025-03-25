"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import logging
import os
from pathlib import Path

import yaml

from benchmarking.algorithms.benchmarking_knn import BenchmarkingKNNAlgorithm
from benchmarking.algorithms.benchmarking_classification import BenchmarkingClassificationAlgorithm
from benchmarking.generator.generator import VerboseSafeDumper
from benchmarking.scrubber.benchmarking_linear_scrubber import BenchmarkingLinearScrubber
from benchmarking.worker.common.statistics_calculation import StatisticsCalculation
from benchmarking.worker.common.threshold_calculation import ThresholdCalculation
from benchmarking.worker.worker import Worker


class OptimalThresholdWorker(Worker):
    def __init__(
        self,
        cpd_algorithm: BenchmarkingKNNAlgorithm | BenchmarkingClassificationAlgorithm,
        optimal_values_storage_path: Path,
        window_length: int,
        shift_factor: float,
        significance_level: float,
        sl_delta: float,
    ) -> None:
        self.__cpd_algorithm = cpd_algorithm
        self.__optimal_value_storage_path = optimal_values_storage_path
        self.__window_length = window_length
        self.__shift_factor = shift_factor
        self.__significance_level = significance_level
        self.__sl_delta = sl_delta

        self.__threshold: float | None = None

    @property
    def threshold(self) -> float | None:
        return self.__threshold

    def run(
        self,
        dataset_path: Path | None,
        results_path: Path,
    ) -> None:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        assert dataset_path is not None, "dataset_path should not be None"

        results_path.mkdir(parents=True, exist_ok=True)
        if not os.listdir(results_path):
            StatisticsCalculation.calculate_statistics(
                self.__cpd_algorithm, self.__window_length, self.__shift_factor, dataset_path, results_path
            )

        scrubber_metaparams = {"type": "linear",
                               "window_length": str(self.__window_length),
                               "shift_factor": str(self.__shift_factor)}
        alg_metaparams = self.__cpd_algorithm.get_metaparameters()

        threshold = ThresholdCalculation.calculate_threshold(
            self.__significance_level,
            1.0,
            results_path,
            self.__sl_delta,
        )

        self.__threshold = threshold

        with open(dataset_path / "config.yaml") as stream:
            loaded_config = yaml.safe_load(stream)

        result_info = [
            {
                "config": {"algorithm": alg_metaparams, "scrubber": scrubber_metaparams},
                "distr_config": loaded_config[0],
                "optimal_values": {"threshold": threshold},
            }
        ]

        with open(self.__optimal_value_storage_path, "a") as outfile:
            yaml.dump(result_info, outfile, default_flow_style=False, sort_keys=False, Dumper=VerboseSafeDumper)
