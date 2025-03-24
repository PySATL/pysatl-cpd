from collections.abc import Callable, Iterable
from enum import Enum
from typing import Any

import numpy
import numpy as np

from .abstract_algorithm import Algorithm
from .graph.builders.matrix import AdjacencyMatrixBuilder
from .graph.graph_cpd import GraphCPD


class Criterion(Enum):
    """
    Attributes:
        Criterion.DEFAULT: Standard algorithm with Graphcpd
        Criterion.TRIANGLE: The criterion based on the number of triangles
    """

    DEFAULT = "default"
    TRIANGLE = "triangle"


class GraphAlgorithm(Algorithm):
    def __init__(
        self, compare_func: Callable[[Any, Any], bool], threshold: float, criterion: Criterion = Criterion.DEFAULT
    ):
        """
        :param compare_func: Comparison function for building a graph.
        :param threshold: Threshold for detecting disorder.
        :param criterion: The chosen criterion for detecting disorder.
        """
        self.compare = compare_func
        self.threshold = threshold
        self.criterion = criterion

    def localize(self, window: Iterable[float | numpy.float64]) -> list[int]:
        if self.criterion == Criterion.DEFAULT:
            cp = self._find_changepoint_default(window)
            return cp
        else:
            raise NotImplementedError(f"Критерий {self.criterion} не поддерживает локализацию")

    def detect(self, window: Iterable[float | numpy.float64]) -> int:
        if self.criterion == Criterion.DEFAULT:
            cp = self._find_changepoint_default(window)
            return len(cp)
        elif self.criterion == Criterion.TRIANGLE:
            return 1 if self._find_changepoint_triangle(window) else 0
        else:
            raise NotImplementedError(f"Критерий {self.criterion} не реализован.")

    def _find_changepoint_default(self, window: Iterable[float | numpy.float64]) -> list[int]:
        """
        The standard method for detecting disorder points using Graphcpd.
        """
        builder = AdjacencyMatrixBuilder(window, self.compare)
        graph = builder.build_graph()
        cpd = GraphCPD(graph)
        num_cpd: list[int] = cpd.find_changepoint(self.threshold)
        return num_cpd

    @staticmethod
    def _normalize(data):
        """
        Normalizes data: leads to standard normal distribution
        """
        data = np.array(data)
        mean_val = np.mean(data)
        std_val = np.std(data)
        return (data - mean_val) / std_val

    @staticmethod
    def _find_changepoint_triangle(data: Iterable[float | numpy.float64]):
        data_standart = GraphAlgorithm._normalize(data)
        const = 0.3
        graph = AdjacencyMatrixBuilder(data_standart, lambda a, b: abs(a - b) <= const).build_graph()
        triangle_count = graph.sum_triangle()
        upper_bound = 1264
        lower_bound = 535
        return triangle_count > lower_bound or triangle_count < upper_bound
