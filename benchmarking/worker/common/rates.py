"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from benchmarking.worker.common.utils import Utils
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.istatistic_test import TestStatistic


class Rates:
    @staticmethod
    def false_negative_rate(
        change_point_i: int,
        statistics: list[float],
        test_statistic: TestStatistic,
        window_size: int,
        interval_length: int,
    ):
        if change_point_i < 0:
            return 0

        change_points = Utils.get_change_points(statistics, test_statistic, window_size)

        true_positives = list(
            filter(lambda x: change_point_i - interval_length <= x <= change_point_i + interval_length, change_points)
        )
        false_negatives = 0 if true_positives else 1

        return false_negatives

    @staticmethod
    def true_positive_rate(
        change_point_i: int,
        statistics: list[float],
        test_statistic: TestStatistic,
        window_size: int,
        interval_length: int,
    ):
        return 1 - Rates.false_negative_rate(change_point_i, statistics, test_statistic, window_size, interval_length)

    @staticmethod
    def false_positive_rate(
        change_point_i: int,
        statistics: list[float],
        test_statistic: TestStatistic,
        window_size: int,
        interval_length: int,
    ):
        data_length = len(statistics)
        change_points = Utils.get_change_points(statistics, test_statistic, window_size)
        overall_count = data_length // (2 * interval_length)

        start = (change_point_i - interval_length) % (2 * interval_length) if change_point_i >= 0 else 0
        predicted_positives = 1 if change_points[:start] else 0
        while start <= data_length:
            predicted_positives += (
                1 if list(filter(lambda x: start <= x < start + 2 * interval_length, change_points)) else 0
            )
            start += 2 * interval_length + 1  # TODO: Think over 1 addition

        true_positives = (
            1
            if change_point_i >= 0
            and list(
                filter(
                    lambda x: change_point_i - interval_length <= x <= change_point_i + interval_length, change_points
                )
            )
            else 0
        )
        false_positives = predicted_positives - true_positives
        negatives = overall_count - 1

        return false_positives / negatives

    @staticmethod
    def true_negative_rate(
        change_point_i: int,
        statistics: list[float],
        test_statistic: TestStatistic,
        window_size: int,
        interval_length: int,
    ):
        return 1 - Rates.false_positive_rate(
            change_point_i, statistics, test_statistic, window_size, interval_length
        )
