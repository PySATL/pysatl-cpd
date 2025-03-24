from typing import Any

import numpy as np
from numpy import dtype, ndarray

from pysatl_cpd.core.algorithms.graph.abstracts.igraph import IGraph


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
        """
        Counts the number of edges between two sets of nodes in a graph.

        Specifically, it counts the edges from nodes with indices less than `thao`
        to nodes with indices greater than or equal to `thao`.

        :param thao: Time index that separates the two sets of nodes.
        :return: The number of edges between the two sets of nodes.
        """
        count_edges = 0
        for node_before in range(thao):
            for node_after in range(thao, self.len):
                if self.mtx[node_before, node_after] == 1:
                    count_edges += 1
        return count_edges

    def sum_of_squares_of_degrees_of_nodes(self) -> int:
        """
        Calculate the sum of squares of degrees of all nodes.

        :return: The sum of squares of degrees of all nodes.
        """
        sum_squares = 0
        for node_1 in range(0, self.len):
            node_degree = 0
            for node_2 in range(0, self.len):
                if self.mtx[node_1, node_2] == 1:
                    node_degree += 1
            node_degree = node_degree**2
            sum_squares += node_degree
        return sum_squares

    def sum_triangle(self) -> int:
        """
        Calculate the number of triangles in the graph (or a subgraph).

        :return: The number of triangles in the graph (or subgraph).
        """

        mtx = self.mtx

        a2 = np.dot(mtx, mtx)
        a3 = np.dot(mtx, a2)
        num_triangles = np.trace(a3) // 6
        if num_triangles == 0:
            return 0
        else:
            return num_triangles

    def node_triangle_frequencies(self) -> np.ndarray:
        """
        Calculate the frequency of each node in triangles.

        :return: An array of node frequencies.
        """
        a2 = np.dot(self.mtx, self.mtx)
        a3 = np.dot(self.mtx, a2)
        frequencies = np.diag(a3) // 2
        return frequencies

    def most_and_least_frequent_nodes(self):
        """
        Find the most and least frequent nodes in triangles.

        :return: A tuple of two arrays, the first containing the most frequent nodes,
                 and the second containing the least frequent nodes.
        """
        freqs = self.node_triangle_frequencies()
        max_freq = np.max(freqs)
        min_freq = np.min(freqs)
        most_frequent = np.where(freqs == max_freq)[0]
        least_frequent = np.where(freqs == min_freq)[0]
        return most_frequent, least_frequent
