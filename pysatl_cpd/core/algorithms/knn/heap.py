"""
Module for implementation of nearest neighbours heap.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import heapq
import typing as tp
from math import isclose

from .abstracts.iobservation import IObservation, Neighbour


class NNHeap:
    """
    The class implementing nearest neighbours heap --- helper abstraction for KNN graph.
    """

    def __init__(
        self,
        size: int,
        metric: tp.Callable[[IObservation, IObservation], float],
        main_observation: IObservation,
        delta: float,
    ) -> None:
        """
        Initializes a new instance of NNHeap.

        :param size: size of the heap.
        :param metric: function for calculating distance between two observations.
        :param main_observation: the central point relative to which the nearest neighbours are sought.
        :param delta: delta for comparing float values of the given observations.
        """
        self.__size = size
        self.__metric = metric
        self.__main_observation = main_observation

        self.__heap: list[Neighbour] = []
        self.__delta = delta

    def build(self, neighbours: list[IObservation]) -> None:
        """
        Builds a nearest neighbour heap relative to the main observation with the given neighbours.

        :param neighbours: list of neighbours.
        """
        for neighbour in neighbours:
            self.__add(neighbour)

    def find_in_heap(self, observation: IObservation) -> bool:
        """
        Checks if the given observation is among the nearest neighbours of the main observation.

        :param observation: observation to test.
        """

        def predicate(x: Neighbour) -> bool:
            return isclose(self.__metric(x.observation, observation), 0.0, rel_tol=self.__delta) and (
                x.observation.time == observation.time
            )

        return any(predicate(i) for i in self.__heap)

    def get_neighbours_indices(self) -> list[int]:
        return [n.observation.time for n in self.__heap]

    def __add(self, observation: IObservation) -> None:
        """
        Adds observation to heap.

        :param observation: observation to add.
        """
        if observation is self.__main_observation:
            return

        # Sign conversion is needed to convert the smallest element heap to the greatest element heap.
        neg_distance = -self.__metric(self.__main_observation, observation)
        neighbour = Neighbour(neg_distance, observation)

        if len(self.__heap) == self.__size and neighbour.distance > self.__heap[0].distance:
            heapq.heapreplace(self.__heap, neighbour)
        elif len(self.__heap) < self.__size:
            heapq.heappush(self.__heap, neighbour)
