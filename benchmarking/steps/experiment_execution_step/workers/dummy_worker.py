"""
Module contains dummy implementation of a worker class for testing and prototyping pipelines.

The DummyWorker provides a minimal Worker implementation that performs no actual computation, serving as a
placeholder for pipeline testing and development purposes.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from benchmarking.custom_types import StorageValues
from benchmarking.logger import cpd_logger
from benchmarking.steps.experiment_execution_step.workers.worker import Worker


class DummyWorker(Worker):
    """Dummy Worker without realisation"""

    def __init__(
        self,
        data_to_return: Optional[dict[str, float]] = None,
        name: str = "Step",
        previous_step_data: Optional[dict[str, Any]] = None,
        config: Optional[Path] = None,
    ) -> None:
        super().__init__(
            name,
            {"b"},
            {"s"},
            {"a"},
            set(),
            previous_step_data,
            config,
        )
        self._data_to_return = data_to_return if data_to_return else dict()

    def run(self, a: int, b: int) -> Iterable[dict[str, StorageValues]]:
        cpd_logger.debug("DummyWorker run method")
        cpd_logger.debug(f"get for execution: a: {a}, b: {b} ")
        return [{"s": a + b}]
