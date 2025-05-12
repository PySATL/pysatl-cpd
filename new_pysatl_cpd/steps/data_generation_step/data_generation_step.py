from pathlib import Path
from typing import Any, Optional

from new_pysatl_cpd.steps.data_generation_step.data_handlers.data_handler import (
    DataHandler,
)
from new_pysatl_cpd.steps.experiment_execution_step.test_execution_step import ExperimentExecutionStep
from new_pysatl_cpd.steps.step import Step


class DataGenerationStep(Step):
    def __init__(
        self,
        data_handler: DataHandler,
        name: str = "Step",
        input_storage_names: Optional[set[str]] = None,
        output_storage_names: Optional[set[str] | dict[str, str]] = None,
        input_step_names: Optional[set[str] | dict[str, str]] = None,
        output_step_names: Optional[set[str] | dict[str, str]] = None,
        config: Optional[Path] = None,
    ):
        super().__init__(
            name,
            input_storage_names,
            output_storage_names,
            input_step_names,
            output_step_names,
            config,
        )
        self.data_handler = data_handler
        self._available_next_classes = [DataGenerationStep, ExperimentExecutionStep]

    def process(self, **kwargs: Any) -> dict[str, float]:
        renamed_step_output = dict()
        renamed_step_input = self._get_step_input(kwargs)

        for data in self.data_handler.get_data(**renamed_step_input):
            renamed_step_output = self._get_step_output(data)
            renamed_storage_output = self._get_storage_output(data)
            if self.saver:
                self.saver(renamed_storage_output)
        return renamed_step_output

    def _validate_storages(self) -> bool:
        return bool(self._saver)
