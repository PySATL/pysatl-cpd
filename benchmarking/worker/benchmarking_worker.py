"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

# from pathlib import Path
# import logging

# from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
# from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
# from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
# from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
# from CPDShell.shell import CPContainer
# from CPDShell.Worker.Common.statistics_calculation import StatisticsCalculation
# from CPDShell.Worker.Common.utils import Utils
# from CPDShell.Worker.worker import Worker


# class CPDBenchmarkWorker(Worker):
#     def __init__(self, expected_change_points: list[int], interval_length: int, logger: logging.Logger) -> None:
#         self.__expected_change_points = expected_change_points
#         self.__interval_length = interval_length

#         self.__average_time: float = 0.0
#         self.__power: float = 0.0
#         self.__logger = logger

#     def run(
#         self,
#         scrubber: LinearScrubber,
#         cpd_algorithm: ClassificationAlgorithm | KNNAlgorithm,
#         dataset_path: Path | None,
#         results_path: Path,
#     ) -> None:
#         """Function for finding change points in window

#         :param window: part of global data for finding change points
#         :return: the number of change points in the window
#         """
#         # TODO: If dataset_path is not None ...
#         assert dataset_path is not None, "Dataset path should not be None"

#         results: list[CPContainer] = StatisticsCalculation.calculate_statistics(cpd_algorithm, scrubber, dataset_path, results_path)

#         power = 0.0
#         significance_level = 0.0
#         avg_delta = 0
#         min_delta = self.__interval_length + 1
#         max_delta = 0
#         overall_tp = 0

#         # Palliative. TODO: generalize
#         for result in results:
#             local_avg_delta = 0
#             local_min_delta = self.__interval_length + 1
#             local_max_delta = -1
#             true_positives = 0

#             for expected_cp in self.__expected_change_points:
#                 deltas = map(lambda actual_cp: abs(actual_cp - expected_cp), result.result)
#                 true_positives_delta = list(filter(lambda x: x <= self.__interval_length, deltas))

#                 if true_positives_delta:
#                     true_positives += 1
#                     local_min_delta = min(local_min_delta, min(true_positives_delta))
#                     local_max_delta = max(local_max_delta, max(true_positives_delta))
#                     local_avg_delta += sum(true_positives_delta) / len(true_positives_delta)

#             if true_positives > 0:
#                 power += true_positives / len(self.__expected_change_points)
#                 overall_tp += 1
#                 avg_delta += local_avg_delta / true_positives
#                 min_delta = min(min_delta, local_min_delta)
#                 max_delta = max(max_delta, local_max_delta)
            
#             # Proposition that expected change points are located far from each other.
#             data_length = len(list(result.data))
#             negatives = data_length // (2 * self.__interval_length) - len(self.__expected_change_points)
#             false_positives = list(filter((lambda actual_cp: all(abs(expected_cp - actual_cp) > self.__interval_length for expected_cp in self.__expected_change_points)), result.result))
#             start = -self.__interval_length - 1
#             filtered_false_positives = []
#             for cp in false_positives:
#                 if start + self.__interval_length < cp:
#                     filtered_false_positives.append(cp)
#                     start = cp

#             false_positive_rate = len(filtered_false_positives) / negatives
#             significance_level += false_positive_rate

#         significance_level = significance_level / len(results)
#         avg_delta = avg_delta / overall_tp
#         self.__power = power / len(results)
#         self.__average_time = sum(result.time_sec for result in results) / len(results)
#         self.__logger.info(f"Power: {self.__power}")
#         self.__logger.info(f"Significance level: {significance_level}")
#         self.__logger.info(f"Average time: {self.__average_time}")
#         self.__logger.info(f"Average interval: {avg_delta}")
#         self.__logger.info(f"Minimum interval: {min_delta}")
#         self.__logger.info(f"Maximum interval: {max_delta}")
#         # It should be calculated while report generating. Time, memory and statistics should be saved. Other metrics should be calculated later.
#         # Utils.print_all_change_points(results_path, ThresholdOvercome(self.__threshold), self.__interval_length)


from pathlib import Path
import logging

from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from CPDShell.shell import CPContainer
from benchmarking.worker.common.statistics_calculation import StatisticsCalculation
from benchmarking.worker.common.utils import Utils
from benchmarking.worker.worker import Worker
from benchmarking.algorithms.benchmarking_knn import BenchmarkingKNNAlgorithm
from benchmarking.scrubber.benchmarking_linear_scrubber import BenchmarkingLinearScrubber


class BenchmarkingKNNWorker(Worker):
    def __init__(self, cpd_algorithm: BenchmarkingKNNAlgorithm, scrubber: BenchmarkingLinearScrubber, expected_change_points: list[int]) -> None:
        self.__expected_change_points = expected_change_points
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
        # TODO: If dataset_path is not None ...
        assert dataset_path is not None, "Dataset path should not be None"

        # TODO: Statistics calculation saves all metrics. Should it be the responsibility of worker?
        StatisticsCalculation.calculate_statistics(self.__cpd_algorithm, self.__scrubber, dataset_path, results_path)
