"""
Module contains core change point detection functionality
combining scrubbers and detection algorithms.
"""

__author__ = "Artem Romanyuk, Vladimir Kutuev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.core.algorithms.abstract_algorithm import Algorithm
from pysatl_cpd.core.scrubber.abstract import Scrubber


class CpdCore:
    """Change Point Detection core"""

    def __init__(
        self,
        scrubber: Scrubber,
        algorithm: Algorithm,
    ) -> None:
        """Change Point Detection core algorithm

        :param scrubber: scrubber for dividing data into windows
            and subsequent processing of data windows
            by change point detection algorithms
        :param algorithm: change point detection algorithm
        :return: list of found change points
        """
        self.scrubber = scrubber
        self.algorithm = algorithm

    def localize(self) -> list[tuple[int, int]]:
        """Find change points

        :return: list of change points
        """
        change_points: list[tuple[int, int]] = []
        for window in self.scrubber.__iter__():
            window_change_points = self.algorithm.localize(window.values)
            time = max(window.indices)
            change_points.extend(map(lambda i: (window.indices[i], time), window_change_points))
        return change_points

    def detect(self) -> list[tuple[int, int]]:
        """Count change points

        :return: number of change points
        """
        change_points_count = 0
        for window in self.scrubber.__iter__():
            change_points_count += self.algorithm.detect(window.values)
        return change_points_count
