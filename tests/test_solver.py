import tempfile
from os import walk
from pathlib import Path

import numpy as np
import pytest

from pysatl_cpd.core.algorithms.graph_algorithm import GraphAlgorithm
from pysatl_cpd.core.problem import CpdProblem
from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider
from pysatl_cpd.core.scrubber.linear import LinearScrubber
from pysatl_cpd.cpd_solver import CpdLocalizationResults, CpdResultsAnalyzer, CpdSolver, LabeledCpdData


def custom_comparison(node1, node2):  # TODO: Remove it everywhere
    arg = 1
    return abs(node1 - node2) <= arg


class TestCpdSolver:
    def test_cpd_localization_no_changepoint(self) -> None:
        data = [1, 2, 3, 4]
        problem = CpdProblem(True)
        algorithm = GraphAlgorithm(custom_comparison, 4)
        scrubber = LinearScrubber(ListUnivariateProvider(data))
        solver = CpdSolver(problem, algorithm, scrubber)
        cpd_result = solver.run()
        assert isinstance(cpd_result, CpdLocalizationResults)
        assert cpd_result.result == []
        assert cpd_result.expected_result is None

    def test_cpd_localization_labeled_data(self) -> None:
        data = LabeledCpdData(np.array([1, 2, 3, 4], dtype=np.float64), [4, 5, 6, 7])
        problem = CpdProblem(True)
        algorithm = GraphAlgorithm(custom_comparison, 4)
        solver = CpdSolver(problem, algorithm, (data, LinearScrubber))
        cpd_result = solver.run()
        assert isinstance(cpd_result, CpdLocalizationResults)
        assert cpd_result.result == []
        assert cpd_result.expected_result == [4, 5, 6, 7]
        assert cpd_result.result_diff == [4, 5, 6, 7]


class TestCPDResultsAnalyzer:
    @pytest.mark.parametrize(
        "result1, result2, window, expected",
        [
            ([4, 5, 6, 7], [3, 5, 6], None, (2, 1, 1, 1)),
            ([4, 5, 6, 7], [3, 5, 6], (5, 6), (1, 0, 0, 0)),
            ([4, 5, 6, 7], [3, 5, 6], (0, 100), (2, 97, 2, 1)),
            ([4, 5, 6, 7], [3, 5, 6], (6, 6), (0, 0, 0, 0)),
            ([3, 5, 6, 7], [4, 5, 6], None, (2, 1, 1, 1)),
            ([], [4, 5, 6], None, (0, 0, 0, 2)),
            ([3, 5, 6, 7], [], None, (0, 4, 3, 0)),
        ],
    )
    def test_count_confusion_matrix(self, result1, result2, window, expected):
        assert CpdResultsAnalyzer.count_confusion_matrix(result1, result2, window) == expected

    def test_count_confusion_matrix_exception_case(self):
        with pytest.raises(ValueError):
            CpdResultsAnalyzer.count_confusion_matrix([], [])

    @pytest.mark.parametrize(
        "result1, result2, window, expected",
        [
            ([4, 5, 6, 7], [3, 5, 6], None, 0.6),
            ([4, 5, 6, 7], [3, 5, 6], (5, 6), 1.0),
            ([4, 5, 6, 7], [3, 5, 6], (6, 6), 0.0),
        ],
    )
    def test_count_accuracy(self, result1, result2, window, expected):
        assert CpdResultsAnalyzer.count_accuracy(result1, result2, window) == expected

    @pytest.mark.parametrize(
        "result1, result2, window, expected",
        [
            ([4, 5, 6, 7], [3, 5, 6], None, 2 / 3),
            ([4, 5, 6, 7], [3, 5, 6], (5, 6), 1.0),
            ([4, 5, 6, 7], [3, 5, 6], (6, 6), 0.0),
        ],
    )
    def test_count_precision(self, result1, result2, window, expected):
        assert CpdResultsAnalyzer.count_precision(result1, result2, window) == expected

    @pytest.mark.parametrize(
        "result1, result2, window, expected",
        [
            ([4, 5, 6, 7], [3, 5, 6], None, 2 / 3),
            ([4, 5, 6, 7], [3, 5, 6], (5, 6), 1.0),
            ([4, 5, 6, 7], [3, 5, 6], (6, 6), 0.0),
        ],
    )
    def test_count_recall(self, result1, result2, window, expected):
        assert CpdResultsAnalyzer.count_recall(result1, result2, window) == expected


class TestCpdLocalizationResults:
    data = [np.float64(1)] * 15
    cont_default1 = CpdLocalizationResults(iter(data), [1, 2, 3], [2, 3, 4], 10)
    cont_default2 = CpdLocalizationResults(iter(data), [1, 2, 3, 6, 8], [2, 3, 4, 6], 20)
    cont_no_expected = CpdLocalizationResults(iter(data), [1, 2, 3], None, 5)

    def test_result_diff(self) -> None:
        assert self.cont_default1.result_diff == [1, 4]
        assert self.cont_default2.result_diff == [1, 4, 8]

    def test_result_diff_exception_case(self) -> None:
        with pytest.raises(ValueError):
            print(self.cont_no_expected.result_diff)

    def test_str_cp_container(self) -> None:
        assert (
            str(self.cont_default1)
            == """Located change points: (1;2;3)
Expected change point: (2;3;4)
Difference: (1;4)
Computation time (sec): 10"""
        )

        assert (
            str(self.cont_default2)
            == """Located change points: (1;2;3;6;8)
Expected change point: (2;3;4;6)
Difference: (1;4;8)
Computation time (sec): 20"""
        )

        assert (
            str(self.cont_no_expected)
            == """Located change points: (1;2;3)
Computation time (sec): 5"""
        )

    @pytest.mark.parametrize(
        "data,name",
        (
            (cont_default1, "d_1"),
            (cont_default2, "d_2"),
            (cont_no_expected, "cne"),
        ),
    )
    def test_visualize(self, data, name) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            data.visualize(False, Path(tempdir), name)
            assert [f"{name}.png"] in [file_names for (_, _, file_names) in walk(tempdir)]

    def test_metric_exception_case(self):
        with pytest.raises(ValueError):
            self.cont_no_expected.count_confusion_matrix()
