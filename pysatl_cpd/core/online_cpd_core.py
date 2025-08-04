"""
Module for online-CPD core, which presents access to algorithms as iterators over provdied data.
"""

__author__ = "Alexey Tatyanenko, Vladimir Kutuev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeVar

from pysatl_cpd.core.algorithms import OnlineAlgorithm
from pysatl_cpd.core.scrubber import DataProvider


@dataclass
class OnlineCpdOutput:
    is_change_point: bool = False
    "Whether a change has been detected."
    change_point_function: float = float("nan")
    "The value of the change point function for the observation."
    step_time: int = 0
    "The number of the processed observation."
    real_time: float = float("nan")
    "The time in seconds spent on processing the observation."


T = TypeVar("T")


class OnlineCpdCore:
    """
    Class that presents online CPD-algorithm as detection iterator over the provided data.
    """

    def __init__(
        self, algorithm: OnlineAlgorithm[T], data_provider: DataProvider[T], threshold: float = float("nan")
    ) -> None:
        self.__algorithm = algorithm
        self.__data_provider = data_provider
        self.__threshold = threshold

    def detect(self) -> Iterator[OnlineCpdOutput]:
        """
        Iteratively tries to detect a change point in the provided data.
        :return: whether a change point after processed observation was detected.
        """
        step_start_time = time.perf_counter()
        for step, observation in enumerate(self.__data_provider):
            cp_func = self.__algorithm.process(observation)
            step_finish_time = time.perf_counter()
            was_cp = False
            if cp_func > self.__threshold:
                self.__algorithm.reset()
                was_cp = True
            yield OnlineCpdOutput(was_cp, cp_func, step, step_finish_time - step_start_time)
            step_start_time = step_finish_time
