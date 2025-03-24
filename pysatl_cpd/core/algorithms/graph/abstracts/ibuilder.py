from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.graph.abstracts.igraph import IGraph


class IBuilder(ABC):
    def __init__(
        self, data: Iterable[np.float64] | Iterable[npt.NDArray[np.float64]], compare: Callable[[Any, Any], bool]
    ):
        """
        Initialize the builder with data and a comparison function.

        :param data: List of elements to be used in building the graph.
        :param compare: Callable that takes two elements and returns a boolean indicating
                        if an edge should exist between them.
        """
        self.data = list(data)
        self.compare = compare
        self.num_of_edges: int = 0

    @abstractmethod
    def build_graph(self) -> IGraph:
        """
        Abstract method to build and return a graph representation.

        :return: An instance of IGraph representing the built graph.
        """
        raise NotImplementedError
