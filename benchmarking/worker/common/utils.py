"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.classification.abstracts.istatistic_test import TestStatistic
from pysatl_cpd.labeled_data import LabeledCpdData


class Utils:
    @staticmethod
    def read_float_data(data_path: Path) -> list[float]:
        with open(data_path) as infile:
            data = list(map(float, infile.readlines()))

        return data

    @staticmethod
    def get_change_points(data: list[float], test_statistic: TestStatistic, window_size: int) -> list[int]:
        change_points = []

        for start in range(0, len(data), window_size):
            change_points += list(
                map(lambda x: x + start, test_statistic.get_change_points(data[start : start + window_size]))
            )

        return change_points

    @staticmethod
    def print_all_change_points(statistics_dir: Path, test_statistic: TestStatistic, window_size: int) -> None:
        stats_paths = Utils.get_all_stats_dirs(statistics_dir)

        for stats_path in stats_paths:
            stats = Utils.read_float_data(stats_path)
            change_points = Utils.get_change_points(stats, test_statistic, window_size)
            print(stats_path)
            print(change_points)

    @staticmethod
    def get_all_stats_dirs(statistics_dir: Path) -> list[Path]:
        root_content = os.listdir(statistics_dir)
        sample_paths = []

        if "stats" in root_content and not os.path.isdir(statistics_dir / "stats"):
            return [statistics_dir]

        for name in root_content:
            cur_path = statistics_dir / name

            if os.path.isdir(cur_path):
                sample_paths.extend(Utils.get_all_stats_dirs(cur_path))

        return sample_paths

    @staticmethod
    def read_all_data_from_dir(dataset_dir: Path) -> list[npt.NDArray[np.float64]]:
        samples = Utils.get_all_sample_dirs(dataset_dir)
        dataset = [LabeledCpdData.read_generated_datasets(p[0])[p[1]].raw_data for p in samples]

        return dataset

    @staticmethod
    def get_all_sample_dirs(dataset_dir: Path) -> list[tuple[Path, str]]:
        root_content = os.listdir(dataset_dir)
        sample_paths = []

        for name in root_content:
            cur_path = dataset_dir / name

            if not os.path.isdir(cur_path):
                continue

            if name.startswith("sample"):
                distr_name = os.listdir(cur_path)[0]
                sample_paths.append((cur_path, distr_name))
            else:
                sample_paths.extend(Utils.get_all_sample_dirs(cur_path))

        return sample_paths
