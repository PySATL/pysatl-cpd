"""
Module contains graph-based change point detection algorithm implementation.
"""

__author__ = "Temerlan Akhmetov, Vladimir Kutuev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .abstract_algorithm import Algorithm
from .graph.builders.matrix import AdjacencyMatrixBuilder
from .graph.graph_cpd import GraphCPD


class GraphAlgorithm(Algorithm):
    def __init__(self, compare_func: Callable[[Any, Any], bool], threshold: float):
        self.compare = compare_func
        self.threshold = threshold

    def localize(self, window: npt.NDArray[np.float64]) -> list[int]:
        builder = AdjacencyMatrixBuilder(window, self.compare)
        graph = builder.build_graph()
        cpd = GraphCPD(graph)
        num_cpd: list[int] = cpd.find_changepoint(self.threshold)
        return num_cpd

    def detect(self, window: npt.NDArray[np.float64]) -> int:
        builder = AdjacencyMatrixBuilder(window, self.compare)
        graph = builder.build_graph()
        cpd = GraphCPD(graph)
        num_cpd: list[int] = cpd.find_changepoint(self.threshold)
        return len(num_cpd)
