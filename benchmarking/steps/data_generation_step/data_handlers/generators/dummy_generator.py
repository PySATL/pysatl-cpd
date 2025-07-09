"""
Module contains implementation Dummy generator for example
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from benchmarking.custom_types import StorageValues
from benchmarking.logger import cpd_logger
from benchmarking.steps.data_generation_step.data_handlers.data_handler import DataHandler


class DummyGenerator(DataHandler):
    """Dummy Generator without realisation"""

    def __init__(
        self,
        name: str = "Step",
        config: Optional[Path] = None,
    ) -> None:
        super().__init__(
            name,
            set(),
            set("B"),
            set(),
            set("A"),
            dict(),
            config,
        )
        self._data_to_return: dict[str, StorageValues] = {"A": 1, "B": 2}

    def get_data(self, *args: Any, **kwargs: Any) -> Iterable[dict[str, StorageValues]]:
        cpd_logger.debug("Dummy generator get_data method")
        cpd_logger.info(f"Dummy generator generated: {self._data_to_return}")
        return [self._data_to_return]
