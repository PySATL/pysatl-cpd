from typing import Any

import numpy as np
from numpy import dtype, ndarray

from CPDShell.Core.algorithms.GpraphCPD.abstracts.igraph import IGraph


class GraphMatrix(IGraph):
    def __init__(self, graph: ndarray[Any, dtype], num_of_edges: int):
        """
        Initialize the GraphMatrix with the adjacency matrix and number of edges.

        :param graph: Adjacency matrix representing the graph.
        :param num_of_edges: Number of edges in the graph.
        """
        super().__init__(num_of_edges, len(graph))
        self.mtx = graph

    def __getitem__(self, item):
        """
        Get the row of the adjacency matrix for a given node.

        :param item: Node index.
        :return: Row of the adjacency matrix corresponding to the node.
        """
        return self.mtx[item]

    def check_edges_exist(self, thao: int) -> int:
        count_edges = 0
        for node_before in range(thao):
            for node_after in range(thao, self.len):
                if self.mtx[node_before, node_after] == 1:
                    count_edges += 1
        return count_edges

    def sum_of_squares_of_degrees_of_nodes(self) -> int:
        sum_squares = 0
        for node_1 in range(0, self.len):
            node_degree = 0
            for node_2 in range(0, self.len):
                if self.mtx[node_1, node_2] == 1:
                    node_degree += 1
            node_degree = node_degree**2
            sum_squares += node_degree
        return sum_squares

    def sum_triangle(self, lower_order: int, higher_order: int) -> int:
        # Проверяем, является ли граф временным и задан ли момент времени
        if higher_order is not None and lower_order is not None:
            # Берем слой матрицы для указанного времени t
            # sub_mtx = self.mtx[:time_index, :time_index]
            sub_mtx = self.mtx[lower_order:higher_order, lower_order:higher_order]
        else:
            # Используем обычную матрицу, если временной параметр не задан
            sub_mtx = self.mtx

        # Считаем количество треугольников по матрице mtx_at_t
        a2 = np.dot(sub_mtx, sub_mtx)
        a3 = np.dot(sub_mtx, a2)
        num_triangles = np.trace(a3) // 6
        return num_triangles
