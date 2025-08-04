"""
Module for implementation of Bayesian CPD algorithm localizer selecting the most probable run length.
"""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.bayesian.abstracts import ILocalizer

__all__ = ["ArgmaxLocalizer"]


class ArgmaxLocalizer(ILocalizer):
    """
    A localizer that localizes a change point corresponding with the most probable non-max run length.
    """

    def localize(self, growth_probs: npt.NDArray[np.float64]) -> int:
        """
        Localizes a change point corresponding with the most probable non-max run length.
        :param growth_probs: growth probabilities for run lengths at the time.
        :return: the most probable non-max run length corresponding change point;
        in case of one-element array returns it.
        """
        max_run_length = growth_probs.shape[0]
        assert max_run_length > 0, "Run length distribution should not be empty"

        return 0 if max_run_length == 1 else int(growth_probs[:-1].argmax())
