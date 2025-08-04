"""
Module for online change point detection algorithm's interface.
"""

__author__ = "Alexey Tatyanenko, Vladimir Kutuev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC
from typing import Generic, TypeVar

__all__ = ["OnlineAlgorithm"]

T = TypeVar("T")


class OnlineAlgorithm(ABC, Generic[T]):
    """
    An abstract class representing the interface of an algorithm for the online change point detection.
    """

    def process(self, observation: T) -> float:
        """Method for a step of detection of a change point.

        :param observation: new observation of a time series.
        :return: change point function value.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clean up the internal state of the algorithm after the change has been detected."""
        raise NotImplementedError


class NoResetOnlineAlgorithm(OnlineAlgorithm[T]):
    def reset(self) -> None:
        return
