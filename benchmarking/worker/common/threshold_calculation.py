"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path

from benchmarking.worker.common.utils import Utils
from pysatl_cpd.core.algorithms.classification.test_statistics.threshold_overcome import ThresholdOvercome


class ThresholdCalculation:
    @staticmethod
    def calculate_threshold(
        significance_level: float,
        threshold: float,
        dataset_path: Path,
        delta: float,
    ) -> float:
        """
        :param sample_length: number of statistical values.
        """
        print(dataset_path)
        dataset = Utils.get_all_sample_dirs(dataset_path)

        cur_threshold = threshold
        cur_sig_level = ThresholdCalculation.__calculate_significance_level(dataset, cur_threshold)
        cur_difference = 1.0

        counter = 0
        while abs(significance_level - cur_sig_level) > delta:
            print(cur_threshold)
            if counter == 20:
                break
            counter += 1

            if cur_sig_level > significance_level:
                cur_threshold = cur_threshold + cur_difference
                cur_sig_level = ThresholdCalculation.__calculate_significance_level(dataset, cur_threshold)

                if cur_sig_level < significance_level + delta:
                    cur_difference /= 2.0
            else:
                cur_threshold = cur_threshold - cur_difference
                cur_sig_level = ThresholdCalculation.__calculate_significance_level(dataset, cur_threshold)

                if cur_sig_level > significance_level - delta:
                    cur_difference /= 2.0

        return cur_threshold

    @staticmethod
    def __calculate_significance_level(dataset_path: list[tuple[Path, str]], threshold: float) -> float:
        """
        :param sample_length: number of statistical values.
        :param interval_length: The length of the intervals that are
         atomically examined for the presense of change point.
        """
        fp_count = 0
        overall_count = len(dataset_path)
        test_statistic = ThresholdOvercome(threshold)

        for data_path in dataset_path:
            data = Utils.read_float_data(data_path[0] / data_path[1])
            if test_statistic.get_change_points(data):
                fp_count += 1

        return fp_count / overall_count
