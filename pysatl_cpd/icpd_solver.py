"""
Module contains protocol for solution of change point detection problem and class for representation of its results.
"""

__author__ = "Aleksei Ivanov, Alexey Tatyanenko, Artem Romanyuk, Vladimir Kutuev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from pysatl_cpd.analysis.results_analyzer import CpdResultsAnalyzer
from pysatl_cpd.core import CpdProblem
from pysatl_cpd.core.online_cpd_core import OnlineCpdOutput

T = TypeVar("T")


class CpdLocalizationResults:
    """Container for results of CPD algorithms"""

    def __init__(
        self,
        data: Iterator[T],
        result: list[OnlineCpdOutput],
        expected_result: list[int] | None,
        problem: CpdProblem,
        time_sec: float,
        threshold: float,
    ) -> None:
        """Object constructor

        :param: result: list, containing change points, that were found by CPD algos
        :param: expected_result: list, containing expected change points, if it is needed
        :param: time_sec: a float number, time of CPD algo execution in fractional seconds
        """
        self.__data = data
        self.__result = result
        self.__detected_changepoints: list[int] = list(
            map(lambda r: r.step_time, filter(lambda r: r.is_change_point, self.__result))
        )
        self.__cpf: list[float] = list(map(lambda r: r.change_point_function, self.__result))
        self.__times: list[float] = list(map(lambda r: r.real_time, self.__result))
        self.__expected_result = expected_result
        self.__time_sec = time_sec
        self.__delay = problem.delay
        self.__threshold = threshold

    @property
    def result_diff(self) -> list[int]:
        """method for calculation symmetrical diff between results and expected results (if its granted)

        :return: symmetrical difference between results and expected results
        """
        if self.__expected_result is None:
            raise ValueError("this object is not provided with expected result, thus diff cannot be calculated.")
        first, second = set(self.__detected_changepoints), set(self.__expected_result)
        return sorted(list(first.symmetric_difference(second)))

    def __str__(self) -> str:
        """method for printing results of CPD algo results in a convenient way

        :return: string with brief CPD algo execution results
        """
        cp_results = ";".join(map(str, self.__detected_changepoints))
        method_output = f"Located change points: ({cp_results})\n"
        if self.__expected_result is not None:
            expected_cp_results = ";".join(map(str, self.__expected_result))
            diff = ";".join(map(str, self.result_diff))
            method_output += f"Expected change point: ({expected_cp_results})\n"
            method_output += f"Difference: ({diff})\n"
        method_output += f"Computation time (sec): {round(self.__time_sec, 2)}"
        return method_output

    def count_confusion_matrix(self, window: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
        """method for counting confusion matrix for hypothesis of equality of CPD results and expected
        results on a window

        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: tuple of integers (true-positive, true-negative, false-positive, false-negative)
        """
        if self.__expected_result is None:
            raise ValueError("this object is not provided with expected result, confusion matrix cannot be calculated")
        return CpdResultsAnalyzer.count_confusion_matrix(self.__detected_changepoints, self.__expected_result, window)

    def count_accuracy(self, window: tuple[int, int] | None = None) -> float:
        """method for counting accuracy metric for hypothesis of equality of CPD results and expected
        results on a window

        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: float, accuracy metric
        """
        if self.__expected_result is None:
            raise ValueError("this object is not provided with expected result, accuracy cannot be calculated")
        return CpdResultsAnalyzer.count_accuracy([cp for cp, _ in self.__result], self.__expected_result, window)

    def count_precision(self, window: tuple[int, int] | None = None) -> float:
        """method for counting precision metric for hypothesis of equality of CPD results and expected
        results on a window

        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: float, precision metric
        """
        if self.__expected_result is None:
            raise ValueError("this object is not provided with expected result, precision cannot be calculated")
        return CpdResultsAnalyzer.count_precision([cp for cp, _ in self.__result], self.__expected_result, window)

    def count_recall(self, window: tuple[int, int] | None = None) -> float:
        """method for counting recall metric for hypothesis of equality of CPD results and expected results on a window

        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: float, recall metric
        """
        if self.__expected_result is None:
            raise ValueError("this object is not provided with expected result, recall cannot be calculated")
        return CpdResultsAnalyzer.count_recall([t for _, t in self.__result], self.__expected_result, window)

    def visualize(
        self,
        to_show: bool = True,
        output_directory: Path | None = None,
        name: str = "plot",
    ) -> None:
        """method for building and analyzing graph

        :param to_show: is it necessary to show a graph
        :param output_directory: If necessary, the path to the directory to save the graph
        :param name: A name for the output image with plot
        """

        # Create plot
        fig, (ax_data, ax_cpf, ax_time) = plt.subplots(3, 1, layout="constrained")

        # Visualize raw data
        data: npt.NDArray[np.float64] = np.array(list(self.__data))
        ax_data.set_title("Data")
        ax_data.plot(data)

        # Visualize true change points
        if self.__expected_result is not None:
            alpha = 0.2
            for cp in self.__expected_result:
                ax_data.axvspan(cp, cp + self.__delay, facecolor="blue", alpha=alpha)
        # Visualize detected change points
        ax_data.vlines(
            x=self.__detected_changepoints,
            ymin=data.min(),
            ymax=data.max(),
            colors="orange",
            ls="--",
        )

        # Visualize change point function
        ax_cpf.set_title("Change point function")
        ax_cpf.plot(self.__cpf, color="red")
        # Visualize threshold
        ax_cpf.axhline(self.__threshold, ls=":", color="darkgreen")

        # Visualize steps time
        ax_time.set_title("Time of steps")
        ax_time.plot(self.__times, color="gold")

        if output_directory:
            if not output_directory.exists():
                output_directory.mkdir()
            plt.savefig(output_directory.joinpath(Path(name)))
        if to_show:
            plt.show()


class ICpdSolver(Protocol):
    """
    Protocol, describing interface of CPD problem's solution.
    """

    def run(self) -> CpdLocalizationResults:
        """
        Method for evaluation of CPD problem's solution, executing some CPD-algorithm.
        :return: CpdLocalizationResults object, containing algo result CP and expected CP if needed,
        or number of detected change points.
        """
        ...
