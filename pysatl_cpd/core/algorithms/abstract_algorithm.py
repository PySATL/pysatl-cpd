from typing import Protocol

import numpy as np
import numpy.typing as npt


class Algorithm(Protocol):
    """Abstract class for change point detection algorithms"""

    def detect(self, window: npt.NDArray[np.float64]) -> int:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        ...

    def localize(self, window: npt.NDArray[np.float64]) -> list[int]:
        """Function for finding coordinates of change points in window

        :param window: part of global data for finding change points
        :return: list of window change points
        """
        ...
