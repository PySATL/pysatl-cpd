"""
Module contains class for solving change point detection problem with an online CPD algorithm.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import time

from pysatl_cpd.core import CpdProblem, OnlineCpdCore
from pysatl_cpd.core.algorithms import OnlineAlgorithm
from pysatl_cpd.core.scrubber import DataProvider, LabeledDataProvider
from pysatl_cpd.icpd_solver import CpdLocalizationResults, ICpdSolver
from pysatl_cpd.labeled_data import LabeledCpdData


class OnlineCpdSolver(ICpdSolver):
    """Class, that grants a convenient interface to
    work with online-CPD algorithms"""

    def __init__(
        self,
        scenario: CpdProblem,
        algorithm: OnlineAlgorithm,
        algorithm_input: DataProvider | LabeledCpdData,
    ) -> None:
        """pysatl_cpd object constructor

        :param: scenario: scenario specify
        :param: algorithm: online-CPD algorithm, that will search for change points
        :param: algorithm_input: data provider or labeled data to construct corresponding data provider.
        """
        self._labeled_data: LabeledCpdData | None = None
        self._cpd_core: OnlineCpdCore
        match algorithm_input:
            case LabeledCpdData() as data:
                self._labeled_data = data
                self._cpd_core = OnlineCpdCore(
                    data_provider=LabeledDataProvider(data),
                    algorithm=algorithm,
                )
            case DataProvider() as data_provider:
                self._cpd_core = OnlineCpdCore(
                    data_provider=data_provider,
                    algorithm=algorithm,
                )

        self._scenario = scenario

    def run(self) -> CpdLocalizationResults | int:
        """Execute online-CPD algorithm and return container with its results

        :return: CpdLocalizationResults object, containing algo result CP and expected CP if needed
        """
        time_start = time.perf_counter()
        if not self._scenario.to_localize:
            return sum(self._cpd_core.detect())

        algo_results = [cp for cp in self._cpd_core.localize() if cp is not None]

        time_end = time.perf_counter()
        expected_change_points: list[int] | None = None
        if isinstance(self._labeled_data, LabeledCpdData):
            expected_change_points = self._labeled_data.change_points
        data = iter(self._cpd_core.data_provider)
        return CpdLocalizationResults(data, algo_results, expected_change_points, time_end - time_start)
