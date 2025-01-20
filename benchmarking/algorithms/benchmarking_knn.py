"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import typing as tp
from time import perf_counter
from collections.abc import Iterable

import numpy as np

from CPDShell.Core.algorithms.abstract_algorithm import Algorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.istatistic_test import TestStatistic
from CPDShell.Core.algorithms.KNNCPD.knn_classifier import KNNClassifier
from benchmarking.algorithms.benchmarking_algorithm import BenchmarkingAlgorithm
from benchmarking.benchmarking_info import AlgorithmBenchmarkingInfo
from benchmarking.benchmarking_info import AlgorithmWindowBenchmarkingInfo

class BenchmarkingKNNAlgorithm(BenchmarkingAlgorithm):
    """
    The class implementing change point detection algorithm based on k-NN classifier. Works only with non-constant data.
    """

    def __init__(
        self,
        distance_func: tp.Callable[[float, float], float],
        test_statistic: TestStatistic,
        indent_coeff: float,
        k=7,
        delta: float = 1e-12,
    ) -> None:
        """
        Initializes a new instance of k-NN based change point detection algorithm.

        :param distance_func: function for calculating the distance between two points in time series.
        :param test_statistic: Criterion to separate change points from other points in sample.
        :param indent_coeff: Coefficient for evaluating indent from window borders.
        The indentation is calculated by multiplying the given coefficient by the size of window.
        :param k: number of neighbours in the knn graph relative to each point.
        Default is 7, which is generally the most optimal value (based on the experiments results).
        :param delta: delta for comparing float values of the given observations.
        """
        self.__test_statistic = test_statistic

        self.__shift_coeff = indent_coeff
        self.__classifier = KNNClassifier(distance_func, k, delta)

        self.__change_points: list[int] = []
        self.__change_points_count = 0

        self.__benchmarking_info: AlgorithmBenchmarkingInfo = []

    @property
    def test_statistic(self) -> TestStatistic:
        return self.__test_statistic

    @test_statistic.setter
    def test_statistic(self, test_statistic) -> None:
        self.__test_statistic = test_statistic

    def get_benchmarking_info(self) -> AlgorithmBenchmarkingInfo:
        current_benchmarking_info = self.__benchmarking_info
        self.__benchmarking_info = []
        return current_benchmarking_info

    def detect(self, window: Iterable[float | np.float64]) -> int:
        """Finds change points in window.

        :param window: part of global data for finding change points.
        :return: the number of change points in the window.
        """
        self.__benchmarking_info.append(self.__process_data(window))
        return self.__change_points_count

    def localize(self, window: Iterable[float | np.float64]) -> list[int]:
        """Finds coordinates of change points (localizes them) in window.

        :param window: part of global data for finding change points.
        :return: list of window change points.
        """
        self.__benchmarking_info.append(self.__process_data(window))
        return self.__change_points.copy()

    def __process_data(self, window: Iterable[float | np.float64]) -> AlgorithmWindowBenchmarkingInfo:
        """
        Processes a window of data to detect/localize all change points depending on working mode.

        :param window: part of global data for change points analysis.
        """
        time_start = perf_counter()

        sample = list(window)
        sample_size = len(sample)
        if sample_size == 0:
            return

        self.__classifier.classify(window)

        # Examining each point.
        # Boundaries are always change points.
        first_point = int(sample_size * self.__shift_coeff)
        last_point = int(sample_size * (1 - self.__shift_coeff))
        assessments = []

        for time in range(first_point, last_point):
            quality = self.__classifier.assess_barrier(time)
            assessments.append(quality)

        time_end = perf_counter()
        change_points = self.__test_statistic.get_change_points(assessments)

        # Shifting change points coordinates according to their place in window.
        self.__change_points = list(map(lambda x: x + first_point, change_points))
        self.__change_points_count = len(change_points)

        return AlgorithmWindowBenchmarkingInfo(assessments, time_end - time_start)
