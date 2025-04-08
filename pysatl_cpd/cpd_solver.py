"""
Module contains class for solving change point detection problem.
"""

__author__ = "Aleksei Ivanov, Artem Romanyuk, Vladimir Kutuev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import time

from .core.algorithms.abstract_algorithm import Algorithm
from .core.cpd_core import CpdCore
from .core.problem import CpdProblem
from .core.scrubber.abstract import Scrubber
from .core.scrubber.data_providers import LabeledDataProvider
from .icpd_solver import CpdLocalizationResults, ICpdSolver
from .labeled_data import LabeledCpdData


class CpdSolver(ICpdSolver):
    """Class, that grants a convenient interface to
    work with CPD algorithms"""

    def __init__(
        self,
        scenario: CpdProblem,
        algorithm: Algorithm,
        algorithm_input: Scrubber | tuple[LabeledCpdData, type[Scrubber]],
    ) -> None:
        """pysatl_cpd object constructor

        :param: scenario: scenario specify
        :param: algorithm: CPD algorithm, that will search for change points
        :param: scrubber: scrubber object for splitting data into parts
        """
        self._labeled_data: LabeledCpdData | None = None
        self._cpd_core: CpdCore
        match algorithm_input:
            case Scrubber() as scrubber:
                self._cpd_core = CpdCore(scrubber, algorithm)
            case (data, scrubber_type):
                self._labeled_data = data
                self._cpd_core = CpdCore(scrubber_type(LabeledDataProvider(data)), algorithm)

        self._scenario = scenario

    def run(self) -> CpdLocalizationResults | int:
        """Execute CPD algorithm and return container with its results

        :return: CpdLocalizationResults object, containing algo result CP and expected CP if needed,
        or number of detected change points.
        """
        time_start = time.perf_counter()
        if not self._scenario.to_localize:
            return self._cpd_core.detect()
        algo_results = self._cpd_core.localize()
        time_end = time.perf_counter()
        expected_change_points: list[int] | None = None
        if isinstance(self._labeled_data, LabeledCpdData):
            expected_change_points = self._labeled_data.change_points
        data = self._cpd_core.scrubber.data
        return CpdLocalizationResults(data, algo_results, expected_change_points, time_end - time_start)
