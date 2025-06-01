from pathlib import Path
from typing import Any, Optional

from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.steps.data_generation_step.data_handlers.data_handler import (
    DataHandler,
)
from new_pysatl_cpd.steps.experiment_execution_step.experiment_execution_step import ExperimentExecutionStep
from new_pysatl_cpd.steps.step import Step


class DataGenerationStep(Step):
    """Concrete step implementation for generating and storing data in a pipeline.

    This step specializes in data generation operations using a provided DataHandler,
    with results automatically saved to storage if a saver is configured.

    :param data_handler: Component responsible for actual data generation logic
    :param name: Human-readable name for this step (default: "Step")
    :param input_storage_names: Required input storage fields (set or dict for renaming)
    :param output_storage_names: Output storage fields (set or dict for renaming)
    :param input_step_names: Required input fields from previous steps (set or dict for renaming)
    :param output_step_names: Output fields for next steps (set or dict for renaming)
    :param config: Path to configuration file

    :ivar data_handler: Data generation component
    :ivar _available_next_classes: Allowed subsequent step types

    .. rubric:: Behavior

    1. Generates data using configured DataHandler
    2. Automatically saves generated data if saver is configured
    3. Validates storage requirements before execution
    4. Can be chained with other DataGenerationSteps or ExperimentExecutionSteps
    """

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
        self._set_storage_data_from_processor(self.data_handler)

    def process(self, *args: Any, **kwargs: Any) -> dict[str, dict[Any, Any]]:
        """Generate and store data using the configured DataHandler.

        :param kwargs: Input parameters for data generation, including:
                      - Storage data (accessed via input_storage_names)
                      - Step metadata (accessed via input_step_names)
        :return: Dictionary of generated data metrics

        .. note::
            - Processes data through the DataHandler in chunks
            - Automatically handles field renaming if output mappings are provided
            - Only saves data if saver is configured (no error if missing)
        """
        renamed_step_output = dict()
        renamed_step_input = self._get_step_input(kwargs)

        for data in self.data_handler.get_data(**renamed_step_input):
            renamed_step_output = self._get_step_output(data)
            renamed_storage_output = self._get_storage_output(data)
            if self._saver:
                for key in renamed_storage_output:
                    self._saver(key, renamed_storage_output[key])
                cpd_logger.info(f"{self} saved data to Storage ({renamed_storage_output})")
        return renamed_step_output

    def _validate_storages(self) -> bool:
        """Verify that required storage connections are established.

        :return: True if saver is configured (loader not required for this step)
        """
        return bool(self._saver)
