"""
Module contains abstract base class for data generation/reading components in change point detection pipelines.

The DataHandler class provides core functionality for streaming data chunks through the processing pipeline,
with required interfaces for concrete data generation implementations.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any

from new_pysatl_cpd.custom_types import StorageValues
from new_pysatl_cpd.steps.step_processor import StepProcessor


class DataHandler(StepProcessor):
    """Abstract base class for data generation components in the DataGenerationStep.

    This specialized StepProcessor handles data generation operations, providing
    an interface for streaming data chunks to DataGenerationSteps.

    .. rubric:: Implementation Requirements

    Concrete subclasses must implement:
    1. The get_data() method to provide processing logic
    2. All required StepProcessor methods
    """

    @abstractmethod
    def get_data(self, *args: Any, **kwargs: Any) -> Iterable[dict[str, StorageValues]]:
        """Generate and yield chunks of processed data (must be implemented by subclasses).

        This is the core data production method that should be implemented to:
        - Process input parameters
        - Generate data in manageable chunks
        - Yield dictionaries of processed results

        :param kwargs: Input parameters for data generation, typically including:
                       - Runtime settings
        :return: Generator yielding dictionaries of processed data where:
                 - Keys are output field names (strings) (e.g. normal_dist)
                 - dicts are containers for generated data (e.g. {1: 1, 2: 20, 3: 40})

        .. note::
            - Implementations should yield rather than return all data at once
            - Each yielded dictionary should contain a complete set of metrics
              for one data chunk
            - The method should clean up resources when generation completes
        """
        ...
