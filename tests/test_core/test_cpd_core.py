import pytest

from pysatl_cpd.core.algorithms.graph_algorithm import GraphAlgorithm
from pysatl_cpd.core.cpd_core import CpdCore
from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider
from pysatl_cpd.core.scrubber.linear import LinearScrubber


def custom_comparison(node1, node2):
    arg = 5
    return abs(node1 - node2) <= arg


class TestCPDCore:
    @pytest.mark.parametrize(
        "data,alg_class,alg_param,expected",
        (
            (
                [50, 55, 60, 48, 52, 70, 75, 80, 90, 85, 95, 100, 50],
                GraphAlgorithm,
                (custom_comparison, 1.5),
                [5],
            ),
        ),
    )
    def test_run(self, data, alg_class, alg_param, expected):
        scrubber = LinearScrubber(ListUnivariateProvider(data))
        algorithm = alg_class(*alg_param)

        core = CpdCore(scrubber, algorithm)
        assert core.localize() == expected
