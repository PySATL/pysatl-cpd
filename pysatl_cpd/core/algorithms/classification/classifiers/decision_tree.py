"""
Module for implementation of decision tree classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from typing import cast

import numpy as np
import numpy.typing as npt
import sklearn.tree as sk

from pysatl_cpd.core.algorithms.classification.abstracts.iclassifier import Classifier

import matplotlib.pyplot as plt

class DecisionTreeClassifier(Classifier):
    """
    The class implementing decision tree classifier for cpd.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance of decision tree classifier for cpd.
        """
        self.__model: sk.DecisionTreeClassifier | None = None

<<<<<<< HEAD:pysatl_cpd/core/algorithms/classification/classifiers/decision_tree.py
    def train(self, sample: npt.NDArray[np.float64], barrier: int) -> None:
=======
    def train(self, sample: np.ndarray, barrier: int) -> None:
>>>>>>> bb8d211 (fix: typing; wip: performance benchmarking):CPDShell/Core/algorithms/ClassificationBasedCPD/classifiers/decision_tree/decision_tree_classifier.py
        """Trains classifier on the given sample.

        :param sample: sample for training classifier.
        :param barrier: index of observation that splits the given sample.
        """
        classes = np.array([0 if i <= barrier else 1 for i in range(len(sample))])
<<<<<<< HEAD:pysatl_cpd/core/algorithms/classification/classifiers/decision_tree.py
        self.__model = sk.DecisionTreeClassifier()
        self.__model.fit(sample, classes)

    def predict(self, sample: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
=======
        self.__model = sk.DecisionTreeClassifier(min_samples_leaf=1)
        tree = self.__model.fit(sample, classes)
        # if barrier == 12:
        #     sk.plot_tree(tree)
        #     plt.show()

    def predict(self, sample: np.ndarray) -> np.ndarray:
>>>>>>> bb8d211 (fix: typing; wip: performance benchmarking):CPDShell/Core/algorithms/ClassificationBasedCPD/classifiers/decision_tree/decision_tree_classifier.py
        """Classifies observations in the given sample based on training with barrier.

        :param sample: sample to classify.
        """
<<<<<<< HEAD:pysatl_cpd/core/algorithms/classification/classifiers/decision_tree.py
        assert self.__model is not None
        return cast(npt.NDArray[np.intp], self.__model.predict(sample))
=======
        assert self.__model is not None, "Classifier should not be None."

        return self.__model.predict(sample)
>>>>>>> bb8d211 (fix: typing; wip: performance benchmarking):CPDShell/Core/algorithms/ClassificationBasedCPD/classifiers/decision_tree/decision_tree_classifier.py
