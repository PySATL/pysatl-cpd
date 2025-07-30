"""
Module containing adjacency matrix builder for graph-based change point detection.
"""

__author__ = " Temerlan Akhmetov, Vladimir Kutuev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.graph.abstracts.ibuilder import IBuilder
from pysatl_cpd.core.algorithms.graph.abstracts.igraph import IGraph
from pysatl_cpd.core.algorithms.graph.graph_matrix import GraphMatrix


class AdjacencyMatrixBuilder(IBuilder):
    def __init__(
        self,
        data: Iterable[np.float64] | Iterable[npt.NDArray[np.float64]],
        comparing_function: Callable[[Any, Any], bool],
    ):
        super().__init__(data, comparing_function)

    def build_matrix(self) -> npt.NDArray[np.int8]:  # Adjacency Matrix
        """
        Build the adjacency matrix from the provided data.

        :return: A NumPy ndarray representing the adjacency matrix where element [i, j] is 1 if
                 there is an edge between node i and node j, otherwise 0.
        """
        count_edges = 0
        count_nodes = len(self.data)
        adjacency_matrix = np.zeros((count_nodes, count_nodes), dtype=np.int8)

        for i in range(count_nodes):
            for j in range(count_nodes):
                if self.compare(self.data[i], self.data[j]) and (i != j):
                    adjacency_matrix[i, j] = 1
                    count_edges += 1
        self.num_of_edges = count_edges // 2

        return adjacency_matrix

    def build_graph(self) -> IGraph:
        graph = self.build_matrix()
        return GraphMatrix(graph, self.num_of_edges)
