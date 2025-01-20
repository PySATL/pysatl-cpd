"""
Module for abstractions used in heap, needed to clearly distinguish observations made at different times.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass


@dataclass
class ScrubberBenchmarkingInfo:
    window_size: int
    shift_factor: float


@dataclass
class AlgorithmWindowBenchmarkingInfo:
    quality_statistics: list[float]
    time: float


AlgorithmBenchmarkingInfo = list[AlgorithmWindowBenchmarkingInfo]
