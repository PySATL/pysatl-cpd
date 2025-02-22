"""
Module for Classification CPD algorithm's classifier abstract base class.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Classifier(ABC):
    """Classifier's abstract base class."""

    @abstractmethod
<<<<<<< HEAD:pysatl_cpd/core/algorithms/classification/abstracts/iclassifier.py
    def train(self, sample: npt.NDArray[np.float64], barrier: int) -> None:
=======
    def train(self, sample: np.ndarray, barrier: int) -> None:
>>>>>>> bb8d211 (fix: typing; wip: performance benchmarking):CPDShell/Core/algorithms/ClassificationBasedCPD/abstracts/iclassifier.py
        """Trains binary classifier on the given sample.
        The observations before barrier belong to the class 0, after barrier --- to the class 1.

        :param sample: sample for training classifier.
        :param barrier: index of observation that splits the given sample.
        """
        raise NotImplementedError

    @abstractmethod
<<<<<<< HEAD:pysatl_cpd/core/algorithms/classification/abstracts/iclassifier.py
    def predict(self, sample: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
=======
    def predict(self, sample: np.ndarray) -> np.ndarray:
>>>>>>> bb8d211 (fix: typing; wip: performance benchmarking):CPDShell/Core/algorithms/ClassificationBasedCPD/abstracts/iclassifier.py
        """Classifies the elements of a sample into one of two classes, based on training with the barrier.

        :param sample: sample to classify.
        """
        raise NotImplementedError
