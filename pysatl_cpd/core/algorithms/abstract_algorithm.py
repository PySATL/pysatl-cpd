from typing import Protocol

import numpy as np
<<<<<<< HEAD:pysatl_cpd/core/algorithms/abstract_algorithm.py
import numpy.typing as npt
=======
>>>>>>> bb8d211 (fix: typing; wip: performance benchmarking):CPDShell/Core/algorithms/abstract_algorithm.py


class Algorithm(Protocol):
    """Abstract class for change point detection algorithms"""

<<<<<<< HEAD:pysatl_cpd/core/algorithms/abstract_algorithm.py
    def detect(self, window: npt.NDArray[np.float64]) -> int:
=======
    @abstractmethod
    def detect(self, window: MutableSequence[float | np.float64 | np.ndarray]) -> int:
>>>>>>> bb8d211 (fix: typing; wip: performance benchmarking):CPDShell/Core/algorithms/abstract_algorithm.py
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        ...

<<<<<<< HEAD:pysatl_cpd/core/algorithms/abstract_algorithm.py
    def localize(self, window: npt.NDArray[np.float64]) -> list[int]:
=======
    @abstractmethod
    def localize(self, window: MutableSequence[float | np.float64 | np.ndarray]) -> list[int]:
>>>>>>> bb8d211 (fix: typing; wip: performance benchmarking):CPDShell/Core/algorithms/abstract_algorithm.py
        """Function for finding coordinates of change points in window

        :param window: part of global data for finding change points
        :return: list of window change points
        """
        ...
