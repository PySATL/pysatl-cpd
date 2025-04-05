"""
Module for online-CPD core, which presents access to algorithms as iterators over provdied data.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterator

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm
from pysatl_cpd.core.scrubber.data_providers import DataProvider


class OnlineCpdCore:
    """
    Class that presents online CPD-algorithm as detection or localization iterator over the provided data.
    """

    def __init__(self, algorithm: OnlineAlgorithm, data_provider: DataProvider) -> None:
        self.algorithm = algorithm
        self.data_provider = data_provider

    def detect(self) -> Iterator[bool]:
        """
        Iteratively tries to detect a change point in the provided data.
        :return: whether a change point after processed observation was detected.
        """
        for observation in self.data_provider:
            yield self.algorithm.detect(observation)

    def localize(self) -> Iterator[int | None]:
        """
        Iteratively tries to localize a change point in the provided data.
        :return: change point location, if it was successfully localized, or None, otherwise.
        """
        for observation in self.data_provider:
            yield self.algorithm.localize(observation)
