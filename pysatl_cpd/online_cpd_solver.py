"""
Module contains class for solving change point detection problem with an online CPD algorithm.
"""

__author__ = "Alexey Tatyanenko, Vladimir Kutuev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import time
from typing import TypeVar

from pysatl_cpd.core import CpdProblem
from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm
from pysatl_cpd.core.online_cpd_core import OnlineCpdCore
from pysatl_cpd.core.scrubber.data_providers import DataProvider
from pysatl_cpd.icpd_solver import CpdLocalizationResults, ICpdSolver

T = TypeVar("T")


class OnlineCpdSolver(ICpdSolver):
    """Class, that grants a convenient interface to
    work with online-CPD algorithms"""

    def __init__(
        self,
        algorithm: OnlineAlgorithm[T],
        data_provider: DataProvider[T],
        threshold: float,
        problem: CpdProblem,
        change_points: list[int] | None = None,
    ) -> None:
        """pysatl_cpd object constructor

        :param: scenario: scenario specify
        :param: algorithm: online-CPD algorithm, that will search for change points
        :param: algorithm_input: data provider or labeled data to construct corresponding data provider.
        """
        self.__cpd_core = OnlineCpdCore(
            data_provider=data_provider,
            algorithm=algorithm,
            threshold=threshold,
        )
        self.__data_provider = data_provider
        self.__expected_change_points = change_points
        self.__problem = problem
        self.__threshold = threshold

    def run(self) -> CpdLocalizationResults:
        """Execute online-CPD algorithm and return container with its results

        :return: CpdLocalizationResults object, containing algo result CP and expected CP if needed
        """
        time_start = time.perf_counter()

        algo_results = list(self.__cpd_core.detect())

        time_end = time.perf_counter()
        data = iter(self.__data_provider)
        return CpdLocalizationResults(
            data, algo_results, self.__expected_change_points, self.__problem, time_end - time_start, self.__threshold
        )
