"""
Module for Abstract Scrubber description.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from abc import abstractmethod

from pysatl_cpd.core.scrubber.abstract import Scrubber


class BenchmarkingScrubber(Scrubber):
    @abstractmethod
    def get_metaparameters(self) -> dict[str, str]:
        raise NotImplementedError
