from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from new_pysatl_cpd.custom_types import StorageValues
from new_pysatl_cpd.steps.data_generation_step.data_handlers.data_handler import DataHandler
from pysatl_cpd.generator.generator import ScipyDatasetGenerator


class CpdGenerator(DataHandler):
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
