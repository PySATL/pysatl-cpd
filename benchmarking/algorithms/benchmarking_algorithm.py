"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from benchmarking.benchmarking_info import AlgorithmBenchmarkingInfo
from CPDShell.Core.algorithms.abstract_algorithm import Algorithm


class BenchmarkingAlgorithm(Algorithm):
    @abstractmethod
    def get_benchmarking_info(self) -> AlgorithmBenchmarkingInfo:
        raise NotImplementedError

    @abstractmethod
    def get_metaparameters(self) -> dict:
        raise NotImplementedError
