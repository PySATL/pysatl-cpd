"""
Module containing interface for graph-based change point detection algorithms.
"""

__author__ = "Temerlan Akhmetov "
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod

from pysatl_cpd.core.algorithms.graph.abstracts.igraph import IGraph


class IGraphCPD(ABC):
    def __init__(self, graph: IGraph):
        """
        Initialize the IGraphCPD with the given graph.

        :param graph: An instance of IGraph representing the graph.
        """
        self.graph = graph
        self.size = graph.len

    @abstractmethod
    def calculation_e(self, thao: int) -> float:
        """
        Calculate the mathematical expectation (E) using the given formula.

        :param thao: Index dividing the nodes into two sets.
        :return: Calculated expectation value.
        """
        raise NotImplementedError

    @abstractmethod
    def calculation_var(self, thao: int) -> float:
        """
        Calculate the variance using the given formula.

        :param thao: Index dividing the nodes into two sets.
        :return: Calculated variance value.
        """
        raise NotImplementedError

    @abstractmethod
    def calculation_z(self, thao: int) -> float:
        """
        Calculate the Z statistic.

        :param thao: Index dividing the nodes into two sets.
        :return: Calculated Z statistic.
        """
        raise NotImplementedError

    @abstractmethod
    def find_changepoint(self, border: float) -> list[int]:
        """
        Find change points in the data based on the Z statistic.

        :param border: Threshold value for detecting change points.
        :return: List of detected change points.
        """
        raise NotImplementedError
