"""
Module for implementation of random forest classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from typing import cast

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier

from pysatl_cpd.core.algorithms.classification.abstracts.iclassifier import Classifier


class RFClassifier(Classifier):
    """
    The class implementing random forest classifier for cpd.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of RF classifier for cpd.
        """
        self.__model: RandomForestClassifier | None = None

    def train(self, sample: npt.NDArray[np.float64], barrier: int) -> None:
        """Trains classifier on the given sample.

        :param sample: sample for training classifier.
        :param barrier: index of observation that splits the given sample.
        """
        classes = [0 if i <= barrier else 1 for i in range(len(sample))]
        self.__model = RandomForestClassifier()
        self.__model.fit(sample, classes)

    def predict(self, sample: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        """Classifies observations in the given sample based on training with barrier.

        :param sample: sample to classify.
        """
        assert self.__model is not None
        return cast(npt.NDArray[np.intp], self.__model.predict(sample))
