"""
Module contains abstract base class for computational worker units in change point detection pipelines.

The Worker class defines the core interface for processing units that execute computational tasks,
providing requirements for concrete implementations that perform actual change point detection operations.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any

from benchmarking.custom_types import StorageValues
from benchmarking.steps.step_processor import StepProcessor


class Worker(StepProcessor):
    """Abstract base class for executable worker units in the ExperimentExecutionStep.

    This class represents a processing unit that performs computational work and yields
    results incrementally. It extends StepProcessor with execution capabilities.

    .. rubric:: Implementation Requirements
    Concrete subclasses must implement:
    1. The run() method to provide processing logic
    2. All required StepProcessor methods
    """

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Iterable[dict[str, StorageValues]]:
        """Execute the worker's processing logic (must be implemented by subclasses).

        This is the main execution method that should:
        - Process input parameters
        - Perform computational work
        - Yield intermediate or final results

        :param kwargs: Input parameters for execution, typically including:
                      - Input data references
                      - Runtime control parameters
        :return: Generator yielding dicts, containing key and dictionary of processed results where:
                 - key is an output field name (string) (e.g. normal_dist_changepoints)
                 - dict contains processed results (e.g. {1: 123, 42344: 9354})

        .. note::
            - Implementations should yield results incrementally where possible
            - Each yielded dictionary should represent a logical unit of work
            - The method should handle its own resource cleanup
            - For long-running operations, consider implementing progress reporting
              through yielded metrics
        """
        ...
