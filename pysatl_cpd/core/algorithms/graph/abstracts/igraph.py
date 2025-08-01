"""
Module containing interface for graph representations used in change point detection algorithms.
"""

__author__ = "Temerlan Akhmetov "
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod


class IGraph(ABC):
    def __init__(self, num_of_edges: int, len_data: int) -> None:
        """
        Initialize the IGraph with the number of edges and the length of data.

        :param num_of_edges: Number of edges in the graph.
        :param len_data: Number of nodes in the graph.
        """
        self.num_of_edges: int = num_of_edges
        self.len: int = len_data

    @abstractmethod
    def check_edges_exist(self, thao: int) -> int:
        """
        Calculate the number of edges that exist between nodes up to a certain index (thao)
        and nodes from that index to the end.

        :param thao: Index dividing the nodes into two sets.
        :return: Number of edges existing between the two sets of nodes.
        """
        raise NotImplementedError

    @abstractmethod
    def sum_of_squares_of_degrees_of_nodes(self) -> int:
        """
        Calculate the sum of the squares of the degrees of all nodes.

        :return: Sum of the squares of the degrees of the nodes.
        """
        raise NotImplementedError
