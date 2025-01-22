"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path

from benchmarking.worker.common.utils import Utils
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from Experiments.rates import Rates


class ThresholdCalculation:
    @staticmethod
    def calculate_threshold(
        significance_level: float,
        threshold: float,
        sample_length: int,
        window_length: int,
        dataset_path: Path,
        interval_length: int,
        delta: float,
    ) -> float:
        """
        :param sample_length: number of statistical values.
        """
        dataset = Utils.get_all_sample_dirs(dataset_path)

        cur_threshold = threshold
        cur_sig_level = ThresholdCalculation.__calculate_significance_level(
            dataset, cur_threshold, sample_length, window_length, interval_length
        )
        cur_difference = 1.0

        while abs(cur_sig_level - significance_level) > delta:
            print(cur_threshold)
            if cur_sig_level > significance_level:
                cur_threshold = cur_threshold + cur_difference
                cur_sig_level = ThresholdCalculation.__calculate_significance_level(
                    dataset, cur_threshold, sample_length, window_length, interval_length
                )

                if cur_sig_level < significance_level + delta:
                    cur_difference /= 2.0
            else:
                cur_threshold = cur_threshold - cur_difference
                cur_sig_level = ThresholdCalculation.__calculate_significance_level(
                    dataset, cur_threshold, sample_length, window_length, interval_length
                )

                if cur_sig_level > significance_level - delta:
                    cur_difference /= 2.0

        return cur_threshold

    @staticmethod
    def __calculate_significance_level(
        dataset_path: list[tuple[Path, str]],
        threshold: float,
        sample_length: int,
        window_length: int,
        interval_length: int,
    ) -> float:
        """
        :param sample_length: number of statistical values.
        :param interval_length: The length of the intervals that are
         atomically examined for the presense of change point.
        """
        fpr_sum = 0
        overall_count = len(dataset_path)
        test_statistic = ThresholdOvercome(threshold)

        for data_path in dataset_path:
            fpr_sum += Rates.false_positive_rate(
                -1, data_path[0], test_statistic, sample_length, window_length, interval_length
            )

        return fpr_sum / overall_count
