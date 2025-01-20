"""
Module for Abstract Scrubber description.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from abc import abstractmethod

from benchmarking.benchmarking_info import ScrubberBenchmarkingInfo
from CPDShell.Core.scrubber.abstract_scrubber import Scrubber


class BenchmarkingScrubber(Scrubber):
    @abstractmethod
    def get_scrubber_benchmarking_info(self) -> ScrubberBenchmarkingInfo:
        raise NotImplementedError
