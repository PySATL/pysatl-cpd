"""
Module for Classification CPD algorithm's quality metric abstract base class.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class QualityMetric(ABC):
    """Quality metric's abstract base class."""

    @abstractmethod
    def assess_barrier(self, classes: npt.NDArray[np.intp], time: int) -> float:
        """Evaluates quality function based on classificator in the specified point.

        :param classes: Classes of observations, predicted by the classifier.
        :param time: Index of barrier in the given sample to calculate quality.
        :return: Quality assessment.
        """
        raise NotImplementedError
