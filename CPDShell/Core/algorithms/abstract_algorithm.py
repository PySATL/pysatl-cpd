from abc import ABC, abstractmethod
from collections.abc import MutableSequence

import numpy


class Algorithm(ABC):
    """Abstract class for change point detection algorithms"""

    @abstractmethod
    def detect(self, window: MutableSequence[float | numpy.float64 | list[numpy.float64]]) -> int:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        raise NotImplementedError

    @abstractmethod
    def localize(self, window: MutableSequence[float | numpy.float64 | list[numpy.float64]]) -> list[int]:
        """Function for finding coordinates of change points in window

        :param window: part of global data for finding change points
        :return: list of window change points
        """
        raise NotImplementedError
