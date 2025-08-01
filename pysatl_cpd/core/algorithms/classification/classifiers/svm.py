"""
Module for implementation of svm classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import typing as tp

import numpy as np
import numpy.typing as npt
from sklearn.svm import SVC

from pysatl_cpd.core.algorithms.classification.abstracts import IClassifier


class SVMClassifier(IClassifier):
    """
    The class implementing svm classifier for cpd.
    """

    def __init__(
        self,
        kernel: tp.Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
    ) -> None:
        """
        Initializes a new instance of svm classifier for cpd.
        :param kernel: specifies the kernel type to be used in the algorithm. If none is given, 'rbf' will be used.
        """
        self.__kernel: tp.Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = kernel
        self.__model: SVC | None = None

    def train(self, sample: npt.NDArray[np.float64], barrier: int) -> None:
        """Trains classifier on the given sample.

        :param sample: sample for training classifier.
        :param barrier: index of observation that splits the given sample.
        """
        classes = np.array([0 if i <= barrier else 1 for i in range(len(sample))])
        self.__model = SVC(kernel=self.__kernel)
        self.__model.fit(sample, classes)

    def predict(self, sample: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        """Classifies observations in the given sample based on training with barrier.

        :param sample: sample to classify.
        """
        assert self.__model is not None
        return tp.cast(npt.NDArray[np.intp], self.__model.predict(sample))
