"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass

import numpy.typing as npt

@dataclass
class ScrubberBenchmarkingInfo:
    window_size: int
    shift_factor: float


@dataclass
class AlgorithmWindowBenchmarkingInfo:
    quality_statistics: list[float]
    time: float


AlgorithmBenchmarkingInfo = list[AlgorithmWindowBenchmarkingInfo]
