"""
Module for implementation of classifier's quality metric based on accuracy.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.classification.abstracts import IQualityMetric


class Accuracy(IQualityMetric):
    """
    The class implementing quality metric based on accuracy.
    """

    def assess_barrier(self, classes: npt.NDArray[np.intp], time: int) -> float:
        """Evaluates quality function based on classificator in the specified point.

        :param classes: Classes of observations, predicted by the classifier.
        :param time: Index of barrier in the given sample to calculate quality.
        :return: Quality assessment.
        """
        before = classes[:time]
        after = classes[time:]
        before_length = time
        sample_length = len(classes)

        true_positive = float(after.sum())
        true_negative = before_length - float(before.sum())

        return (true_positive + true_negative) / sample_length
