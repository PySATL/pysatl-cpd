import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from .analysis.results_analyzer import CpdResultsAnalyzer
from .core.algorithms.abstract_algorithm import Algorithm
from .core.cpd_core import CpdCore
from .core.problem import CpdProblem
from .core.scrubber.abstract import Scrubber
from .core.scrubber.data_providers import LabeledDataProvider
from .labeled_data import LabeledCpdData


class CpdLocalizationResults:
    """Container for results of CPD algorithms"""

    def __init__(
        self,
        data: Iterator[np.float64] | Iterator[npt.NDArray[np.float64]],
        result: list[int],
        expected_result: list[int] | None,
        time_sec: float,
    ) -> None:
        """Object constructor

        :param: result: list, containing change points, that were found by CPD algos
        :param: expected_result: list, containing expected change points, if it is needed
        :param: time_sec: a float number, time of CPD algo execution in fractional seconds
        """
        self.data = data
        self.result = result
        self.expected_result = expected_result
        self.time_sec = time_sec

    @property
    def result_diff(self) -> list[int]:
        """method for calculation symmetrical diff between results and expected results (if its granted)

        :return: symmetrical difference between results and expected results
        """
        if self.expected_result is None:
            raise ValueError("this object is not provided with expected result, thus diff cannot be calculated.")
        first, second = set(self.result), set(self.expected_result)
        return sorted(list(first.symmetric_difference(second)))

    def __str__(self) -> str:
        """method for printing results of CPD algo results in a convenient way

        :return: string with brief CPD algo execution results
        """
        cp_results = ";".join(map(str, self.result))
        method_output = f"Located change points: ({cp_results})\n"
        if self.expected_result is not None:
            expected_cp_results = ";".join(map(str, self.expected_result))
            diff = ";".join(map(str, self.result_diff))
            method_output += f"Expected change point: ({expected_cp_results})\n"
            method_output += f"Difference: ({diff})\n"
        method_output += f"Computation time (sec): {round(self.time_sec, 2)}"
        return method_output

    def count_confusion_matrix(self, window: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
        """method for counting confusion matrix for hypothesis of equality of CPD results and expected
        results on a window

        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: tuple of integers (true-positive, true-negative, false-positive, false-negative)
        """
        if self.expected_result is None:
            raise ValueError("this object is not provided with expected result, confusion matrix cannot be calculated")
        return CpdResultsAnalyzer.count_confusion_matrix(self.result, self.expected_result, window)

    def count_accuracy(self, window: tuple[int, int] | None = None) -> float:
        """method for counting accuracy metric for hypothesis of equality of CPD results and expected
        results on a window

        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: float, accuracy metric
        """
        if self.expected_result is None:
            raise ValueError("this object is not provided with expected result, accuracy cannot be calculated")
        return CpdResultsAnalyzer.count_accuracy(self.result, self.expected_result, window)

    def count_precision(self, window: tuple[int, int] | None = None) -> float:
        """method for counting precision metric for hypothesis of equality of CPD results and expected
        results on a window

        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: float, precision metric
        """
        if self.expected_result is None:
            raise ValueError("this object is not provided with expected result, precision cannot be calculated")
        return CpdResultsAnalyzer.count_precision(self.result, self.expected_result, window)

    def count_recall(self, window: tuple[int, int] | None = None) -> float:
        """method for counting recall metric for hypothesis of equality of CPD results and expected results on a window

        :param: window: tuple of two indices (start, stop), determines a window for hypothesis

        :return: float, recall metric
        """
        if self.expected_result is None:
            raise ValueError("this object is not provided with expected result, recall cannot be calculated")
        return CpdResultsAnalyzer.count_recall(self.result, self.expected_result, window)

    def visualize(self, to_show: bool = True, output_directory: Path | None = None, name: str = "Graph") -> None:
        """method for building and analyzing graph

        :param to_show: is it necessary to show a graph
        :param output_directory: If necessary, the path to the directory to save the graph
        :param name: If necessary, graph name for saving
        """

        data: npt.NDArray[np.float64] = np.array(list(self.data))
        plt.plot(data)
        if self.expected_result is None:
            plt.vlines(x=self.result, ymin=data.min(), ymax=data.max(), colors="orange", ls="--")
            plt.gca().legend(("data", "detected"))
        else:
            correct, incorrect, undetected = set(), set(), set(self.expected_result)
            for point in self.result:
                if point in self.expected_result:
                    correct.add(point)
                    undetected.remove(point)
                elif point not in undetected:
                    incorrect.add(point)
            plt.vlines(x=list(correct), ymin=data.min(), ymax=data.max(), colors="green", ls="--")
            plt.vlines(x=list(incorrect), ymin=data.min(), ymax=data.max(), colors="red", ls="--")
            plt.vlines(x=list(undetected), ymin=data.min(), ymax=data.max(), colors="orange", ls="--")
            plt.gca().legend(("data", "correct detected", "incorrect detected", "undetected"))
        if output_directory:
            if not output_directory.exists():
                output_directory.mkdir()
            plt.savefig(output_directory.joinpath(Path(name)))
        if to_show:
            plt.show()


class CpdSolver:
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

        :return: CPContainer object, containing algo result CP and expected CP if needed
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
