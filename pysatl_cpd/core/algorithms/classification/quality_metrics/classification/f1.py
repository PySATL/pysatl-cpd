"""
Module for implementation of classifier's quality metric based on F1 score.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.classification.abstracts.iquality_metric import (
    QualityMetric,
)


class F1(QualityMetric):
    """
    The class implementing quality metric based on F1 score.
    """

    def assess_barrier(self, classes: npt.NDArray[np.intp], time: int) -> float:
        """Evaluates quality function based on classificator in the specified point.

        :param classes: Classes of observations, predicted by the classifier.
        :param time: Index of barrier in the given sample to calculate quality.
        :return: Quality assessment.
        """
        before = classes[:time]
        after = classes[time:]
        after_length = len(after)

        true_positive = float(after.sum())
        false_positive = float(before.sum())
        false_negative = after_length - true_positive

        return 2 * true_positive / (2 * true_positive + false_positive + false_negative)
