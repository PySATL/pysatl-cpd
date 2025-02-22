"""
Module for implementation of CPD algorithm based on classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable
from time import perf_counter

import numpy as np

from benchmarking.algorithms.benchmarking_algorithm import BenchmarkingAlgorithm
from benchmarking.benchmarking_info import AlgorithmBenchmarkingInfo, AlgorithmWindowBenchmarkingInfo
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.istatistic_test import TestStatistic
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iclassifier import Classifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iquality_metric import QualityMetric


class BenchmarkingClassificationAlgorithm(BenchmarkingAlgorithm):
    """
    The class implementing change point detection algorithm based on classification.
    """

    def __init__(
        self, classifier: Classifier, quality_metric: QualityMetric, test_statistic: TestStatistic, indent_coeff: float
    ) -> None:
        """
        Initializes a new instance of classification based change point detection algorithm.

        :param classifier: Classifier for sample classification.
        :param quality_metric: Metric to assess independence of the two samples
        resulting from splitting the original sample.
        :param test_statistic: Criterion to separate change points from other points in sample.
        :param indent_coeff: Coefficient for evaluating indent from window borders.
        The indentation is calculated by multiplying the given coefficient by the size of window.
        """
        self.__classifier = classifier
        self.__test_statistic = test_statistic
        self.__quality_metric = quality_metric

        self.__shift_coeff = indent_coeff

        self.__change_points: list[int] = []
        self.__change_points_count = 0

        self.__metaparameters_info = {
            "type": classifier.__class__.__name__,
            "quality_metric": quality_metric.__class__.__name__,
            "indent_coeff": indent_coeff,
        }
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

    def get_metaparameters(self) -> dict:
        return self.__metaparameters_info

    def detect(self, window: MutableSequence[float | np.float64 | np.ndarray]) -> int:
        """Finds change points in window.

        :param window: part of global data for finding change points.
        :return: the number of change points in the window.
        """
        self.__benchmarking_info.append(self.__process_data(window))
        return self.__change_points_count

    def localize(self, window: MutableSequence[float | np.float64 | np.ndarray]) -> list[int]:
        """Finds coordinates of change points (localizes them) in window.

        :param window: part of global data for finding change points.
        :return: list of window change points.
        """
        self.__benchmarking_info.append(self.__process_data(window))
        return self.__change_points.copy()

    def __process_data(self, window: MutableSequence[float | np.float64 | np.ndarray]) -> AlgorithmWindowBenchmarkingInfo:
        """
        Processes a window of data to detect/localize all change points depending on working mode.

        :param window: part of global data for change points analysis.
        """
        time_start = perf_counter()

        sample = list(window)
        sample_size = len(sample)
        if sample_size == 0:
            return

        # Examining each point.
        # Boundaries are always change points.
        first_point = int(sample_size * self.__shift_coeff)
        last_point = int(sample_size * (1 - self.__shift_coeff))
        assessments = []

        for time in range(first_point, last_point):
            train_sample, test_sample = BenchmarkingClassificationAlgorithm.__split_sample(sample)
            self.__classifier.train(train_sample, int(time / 2))
            classes = self.__classifier.predict(test_sample)

            quality = self.__quality_metric.assess_barrier(classes, int(time / 2))
            assessments.append(quality)

        time_end = perf_counter()
        change_points = self.__test_statistic.get_change_points(assessments)

        # Shifting change points coordinates according to their place in window.
        self.__change_points = list(map(lambda x: x + first_point, change_points))
        self.__change_points_count = len(change_points)

        return AlgorithmWindowBenchmarkingInfo(assessments, time_end - time_start)

    # Splits the given sample into train and test samples.
    # Strategy: even elements goes to the train sample; uneven --- to the test sample
    # Soon classification algorithm will be more generalized: the split strategy will be one of the parameters.
    @staticmethod
    def __split_sample(
        sample: MutableSequence[float | np.float64 | np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        train_sample = []
        test_sample = []

        for i, x in enumerate(sample):
            if isinstance(x, np.ndarray):
                new_el = x
            else:
                new_el = [x]
            
            if i % 2 == 0:
                train_sample.append(new_el)
            else:
                test_sample.append(new_el)

        return np.array(train_sample), np.array(test_sample)
