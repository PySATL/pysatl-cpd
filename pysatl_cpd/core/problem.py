"""
Module contains dataclass specification for change point detection problem.
"""

__author__ = "Vladimir Kutuev, Loikov Vladislav"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass


@dataclass
class CpdProblem:
    """Specification of the solving problem."""

    time_delay: float | None = 1
    """The delay is in seconds. If the algorithm detects a change during this delay,
    it is considered that it was detected successfully.
    """
    delay: int | None = 50
    """The delay is in number of steps (observations)."""
