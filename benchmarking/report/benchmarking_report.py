"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path

import yaml

from benchmarking.worker.common.utils import Utils
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome


# User can filter out none values and use only some.
# Actually, Measures can be class with method to incalsulate this work.
class Measures:
    def __init__(self) -> None:
        self.average_overall_time: float | None = None
        self.average_window_time: float | None = None
        self.memory: float | None = None  # TODO: There are different types of memory
        self.power: float | None = None
        self.f1: float | None = None
        self.sl: float | None = None
        self.interval: int | None = None
        self.scrubbing_alg_info: dict[str, dict[str, str]] | None = None

    def filter_out_none(self) -> dict[str, float]:
        return {k: v for k, v in vars(self).items() if v is not None}


class BenchmarkingReport:
    def __init__(self, resultsDir: Path, expected_cp: list[int], threshold: float, interval_length: int) -> None:
        self.__resultsDir = resultsDir
        self.__expected_cp = expected_cp
        self.__theshold = threshold
        self.__interval_length = interval_length

        self.__result: Measures = Measures()
        self.__sample_dirs = Utils.get_all_stats_dirs(resultsDir)

    def add_average_overall_time(self) -> None:
        overall_time = 0

        for sample_dir in self.__sample_dirs:
            with open(sample_dir / "benchmarking_info.yaml") as infile:
                # TODO: Unsure about type.
                loaded_info: dict[str, float] = yaml.safe_load(infile)

            overall_time += loaded_info["overall_time"]

        self.__result.average_overall_time = overall_time / len(self.__sample_dirs)

    def add_average_window_time(self) -> None:
        overall_time = 0

        for sample_dir in self.__sample_dirs:
            with open(sample_dir / "benchmarking_info.yaml") as infile:
                # TODO: Unsure about type.
                loaded_info: dict[str, float] = yaml.safe_load(infile)

            overall_time += loaded_info["average_time"]

        self.__result.average_window_time = overall_time / len(self.__sample_dirs)

    def add_memory(self) -> None:
        raise NotImplementedError

    def add_power(self) -> None:
        power_sum = 0.0

        for sample_dir in self.__sample_dirs:
            stats = Utils.read_float_data(sample_dir / "stats")
            actual_cp = Utils.get_change_points(stats, ThresholdOvercome(self.__theshold), len(stats))

            true_positives = 0

            for exp_cp in self.__expected_cp:
                true_positives_delta = list(
                    filter(lambda act_cp: abs(act_cp - exp_cp) <= self.__interval_length, actual_cp)
                )

                if true_positives_delta:
                    true_positives += 1

            if true_positives > 0:
                power_sum += true_positives / len(self.__expected_cp)

        self.__result.power = power_sum / len(self.__sample_dirs)

    def add_F1(self) -> None:
        raise NotImplementedError

    def add_SL(self) -> None:
        raise NotImplementedError

    def add_interval(self) -> None:
        raise NotImplementedError

    def add_scrubbing_alg_info(self) -> None:
        with open(self.__resultsDir / "config.yaml") as infile:
            # TODO: Unsure about type.
            loaded_info: dict[str, dict[str, str]] = yaml.safe_load(infile)

        self.__result.scrubbing_alg_info = loaded_info

    def get_result(self) -> Measures:
        return self.__result
