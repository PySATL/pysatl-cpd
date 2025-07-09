"""
Module contains concrete implementation of an experiment execution step for change point detection pipelines.

The ExperimentExecutionStep class manages the execution of computational workers, handling data loading from storage,
worker execution, and result saving. It serves as the core component for running change point detection
experiments in the pipeline workflow.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path
from typing import Any, Optional

from benchmarking.custom_types import StorageValues
from benchmarking.steps.experiment_execution_step.workers.worker import Worker
from benchmarking.steps.report_generation_step.report_generation_step import ReportGenerationStep
from benchmarking.steps.step import Step


class ExperimentExecutionStep(Step):
    """Concrete step implementation for executing experimental workflows in a pipeline.

    This step manages the execution of Worker instances, handling their input/output
    data and processing results. It serves as an adapter between the pipeline
    infrastructure and individual Worker implementations.

    :param worker: The worker instance that performs the actual computation
    :param name: Human-readable name for this step (default: "Step")
    :param input_storage_names: Required input storage fields (set or dict for renaming)
    :param output_storage_names: Output storage fields (set or dict for renaming)
    :param input_step_names: Required input fields from previous steps (set or dict for renaming)
    :param output_step_names: Output fields for next steps (set or dict for renaming)
    :param config: Path to configuration file (the config is optional. If available, it will be passed to
    the StepProcessor and processed there. Makes it possible to additionally configure a specific StepProcessor)

    :ivar _worker: The wrapped worker instance
    :ivar _available_next_classes: Allowed subsequent step types

    .. rubric:: Execution Flow

    1. Loads required input data from storage
    2. Executes the worker's run() method with combined inputs
    3. Processes and saves each result chunk
    4. Returns aggregated step outputs

    .. rubric:: Example Usage

    Creating and using an execution step::

        worker = AnalysisWorker()
        step = ExperimentExecutionStep(
            worker=worker,
            name="MaterialAnalysis",
            input_storage_names={"sample_data"},
            output_storage_names={"analysis_results"},
            config=Path("configs/analysis.yaml"),
        )
    """

    def __init__(
        self,
        worker: Worker,
        name: str = "Step",
        input_storage_names: Optional[set[str] | dict[str, str]] = None,
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
        self._worker = worker
        self._available_next_classes = [ExperimentExecutionStep, ReportGenerationStep]
        self._set_storage_data_from_processor(self._worker)

    def process(self, *args: Any, **kwargs: Any) -> dict[str, StorageValues]:
        """Execute the experimental workflow and process results.

        :param kwargs: Input parameters including:
                      - Storage data (accessed via input_storage_names)
                      - Step metadata (accessed via input_step_names)
        :return: Dictionary of processed results

        .. note::
            - Processes worker results incrementally
            - Only saves data if saver is configured
            - Combines storage data and step metadata for worker execution
        """

        if self.loader is None:
            raise ValueError("Storage loader is not initialized")

        load_from_storage_names = (
            self.input_storage_names
            if isinstance(self.input_storage_names, set)
            else set(self.input_storage_names.keys())
        )

        storage_input: dict[str, StorageValues] = self.loader(load_from_storage_names)

        renamed_storage_input = self._get_storage_input(storage_input)
        renamed_step_input = self._get_step_input(kwargs)

        # Maybe result must be iterable, or remove 'for worker_result ...'
        renamed_step_output = dict()

        for worker_result in self._worker.run(**renamed_storage_input, **renamed_step_input):
            renamed_step_output = self._get_step_output(worker_result)
            renamed_storage_output = self._get_storage_output(worker_result)
            if self._saver:
                for key in renamed_storage_output:
                    self._saver(key, renamed_storage_output[key])

        return renamed_step_output

    def _validate_storages(self) -> bool:
        """Verify that required storage connections are established.

        :return: True if both saver and loader are configured
        """
        return bool(self._saver and self._loader)
