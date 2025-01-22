"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from pathlib import Path


class Worker(ABC):
    """Abstract class for worker"""

    @abstractmethod
    def run(
        self,
        dataset_path: Path | None,
        results_path: Path,
    ) -> None:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        raise NotImplementedError
