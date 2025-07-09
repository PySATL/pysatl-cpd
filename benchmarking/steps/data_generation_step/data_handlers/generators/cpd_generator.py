"""
Module contains implementation of a DataHandler.

Basic generator for distributions from previous architecture. Used in example.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from benchmarking.custom_types import StorageValues
from benchmarking.steps.data_generation_step.data_handlers.data_handler import DataHandler
from pysatl_cpd.generator.generator import ScipyDatasetGenerator


class CpdGenerator(DataHandler):
    """Generator, based on ScipyDatasetGenerator, used in example"""

    def __init__(
        self,
        name: str = "Step",
        input_storage_names: Optional[set[str]] = None,
        output_storage_names: Optional[set[str]] = None,
        input_step_names: Optional[set[str]] = None,
        output_step_names: Optional[set[str]] = None,
        previous_step_data: Optional[dict[str, Any]] = None,
        config: Optional[Path] = None,
    ) -> None:
        super().__init__(
            name,
            input_storage_names,
            output_storage_names,
            input_step_names,
            output_step_names,
            previous_step_data,
            config,
        )
        self.generator = ScipyDatasetGenerator()

    def get_data(self, *args: Any, **kwargs: Any) -> Iterable[dict[str, StorageValues]]:
        datasets = self.generator.generate_datasets(self.config)
        for dataset in datasets:
            yield {dataset: datasets[dataset][0]}
