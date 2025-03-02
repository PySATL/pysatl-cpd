"""
Module for implementation of decision tree classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
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

    def train(self, sample: np.ndarray, barrier: int) -> None:
        """Trains classifier on the given sample.

        :param sample: sample for training classifier.
        :param barrier: index of observation that splits the given sample.
        """
        classes = np.array([0 if i <= barrier else 1 for i in range(len(sample))])
        self.__model = sk.DecisionTreeClassifier(min_samples_leaf=1)
        tree = self.__model.fit(sample, classes)
        # if barrier == 12:
        #     sk.plot_tree(tree)
        #     plt.show()

    def predict(self, sample: np.ndarray) -> np.ndarray:
        """Classifies observations in the given sample based on training with barrier.

        :param sample: sample to classify.
        """
        assert self.__model is not None, "Classifier should not be None."

        return self.__model.predict(sample)
