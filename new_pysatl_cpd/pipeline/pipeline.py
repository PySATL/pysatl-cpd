"""
Module contains implementation of a pipeline for sequential execution of data processing steps, including data
generation, experiment execution and report generation.

The pipeline handles step compatibility verification, data storage management and metadata passing between steps.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, Optional

from new_pysatl_cpd.logger import cpd_logger, log_exceptions
from new_pysatl_cpd.steps.data_generation_step.data_generation_step import DataGenerationStep
from new_pysatl_cpd.steps.experiment_execution_step.experiment_execution_step import ExperimentExecutionStep
from new_pysatl_cpd.steps.report_generation_step.report_generation_step import ReportGenerationStep
from new_pysatl_cpd.steps.step import Step
from new_pysatl_cpd.storages.loaders.csv_loader.csv_loader import LoaderCSV
from new_pysatl_cpd.storages.loaders.loader import Loader
from new_pysatl_cpd.storages.savers.csv_saver.csv_saver import SaverCSV
from new_pysatl_cpd.storages.savers.saver import Saver


class Pipeline:
    """Main pipeline class for executing a sequence of processing steps.

    The pipeline manages the execution flow between different types of steps,
    handles data storage requirements, and ensures step compatibility.

    :param steps: Ordered list of steps to execute in the pipeline

    :ivar _generated_data_storage_fields: Set of field names for generated data storage
    :ivar _result_storage_fields: Set of field names for result storage
    :ivar _meta_data: Dictionary for storing metadata between steps
    :ivar _generated_data_saver: Saver instance for generated data
    :ivar _generated_data_loader: Loader instance for generated data
    :ivar _result_saver: Saver instance for results
    :ivar _result_loader: Loader instance for results

    .. rubric:: Usage Example

    Typical pipeline construction::

        steps = [DataGenerationStep(...), ExperimentExecutionStep(...), ReportGenerationStep(...)]
        pipeline = Pipeline(steps)
        pipeline.run()

    .. note::
        The pipeline automatically configures itself during initialization
        by calling `config_pipeline` method.
    """

    def __init__(
            self,
            steps: list[Step],
            *,
            generation_saver: Optional[Saver] = None,
            generation_loader: Optional[Loader] = None,
            result_saver: Optional[Saver] = None,
            result_loader: Optional[Loader] = None,
    ):
        """Initialize the pipeline with processing steps."""
        self.steps = steps
        self._generated_data_storage_fields: set[str] = set()
        self._result_storage_fields: set[str] = set()
        self._meta_data: dict[str, dict[Any, Any]] = dict()
        self._generated_data_saver: Optional[Saver] = (
            generation_saver if generation_saver is not None else SaverCSV("generation")
        )
        self._generated_data_loader: Optional[Loader] = (
            generation_loader if generation_loader is not None else LoaderCSV("generation")
        )
        self._result_saver: Optional[Saver] = result_saver if result_saver is not None else SaverCSV("results")
        self._result_loader: Optional[Loader] = result_loader if result_loader is not None else LoaderCSV("results")
        self.config_pipeline()

    def _check_two_steps(self, step_1: Step, step_2: Step) -> None:
        """Verify compatibility between two consecutive steps.

        :param step_1: The preceding step in the pipeline
        :param step_2: The following step in the pipeline
        :raises ValueError: If step types are unexpected
        :raises KeyError: If required storage fields or metadata are missing
        """

        if isinstance(step_1, DataGenerationStep):
            storage_fields = self._generated_data_storage_fields
        elif isinstance(step_1, (ExperimentExecutionStep, ReportGenerationStep)):
            storage_fields = self._result_storage_fields
        else:
            raise ValueError(
                f"{step_1} is unexpected Step (not one of DataGenerationStep,"
                f" ExperimentExecutionStep, ReportGenerationStep)"
            )

        if isinstance(step_1.output_storage_names, set):
            storage_fields = storage_fields.union(step_1.output_storage_names)
        else:
            storage_fields = storage_fields.union(step_1.output_storage_names.values())

        output_step_names = (
            step_1.output_step_names if isinstance(step_1.output_step_names, set) else step_1.output_step_names.values()
        )
        for key in output_step_names:
            self._meta_data[key] = {}

        # Check input from storage
        input_storage_names = (
            step_2.input_storage_names
            if isinstance(step_2.input_storage_names, set)
            else set(step_2.input_storage_names.keys())
        )
        if not input_storage_names.issubset(storage_fields):
            cur_input_storage_names = set()
            if isinstance(step_2.input_storage_names, set):
                cur_input_storage_names = step_2.input_storage_names
            else:
                cur_input_storage_names = set(step_2.input_storage_names.keys())

            missed_fields = cur_input_storage_names - storage_fields
            raise KeyError(
                f" For {step_2} to work, there must be values {missed_fields} in the storage."
                f" Check if this fields are created accurately in the previous steps."
                f" (If these fields are not needed for {step_2} to work,"
                f" then remove them from the input_storage_names field)."
                f" current fields in storage (for this step): {storage_fields}."
                f" Maybe you need to rename data in step."
            )

        # Check input from step
        input_step_names = (
            step_2.input_step_names if isinstance(step_2.input_step_names, set) else set(step_2.input_step_names.keys())
        )
        cpd_logger.debug(f"input_step_names: {input_step_names}, current meta data {self._meta_data.keys()}")
        if not input_step_names.issubset(self._meta_data.keys()):
            missed_fields = input_step_names - self._meta_data.keys()
            raise KeyError(
                f" For {step_2} to work, there must be values {missed_fields} returned from previous steps"
                f" as meta data."
                f" Check if this fields are created accurately in the previous steps."
                f" (If these fields are not needed for {step_2} to work,"
                f" then remove them from the input_step_names field)"
                f" current metadata: {set(self._meta_data.keys())})"
                f" Maybe you need to rename data in step."
            )
        if isinstance(step_1, DataGenerationStep):
            self._generated_data_storage_fields = storage_fields
        elif isinstance(step_1, (ExperimentExecutionStep, ReportGenerationStep)):
            self._result_storage_fields = storage_fields

    def _setup_step_storage(self, step: Step) -> None:
        """Configure storage handlers for a specific step.

        :param step: The step to configure
        :raises ValueError: If step type is unexpected
        """
        cpd_logger.debug(f"{step} Storages: START SETUP")
        # TODO Rework (Open-Close problem) NEW: Step method _get_storages -> Optional[Saver], Optional[Loader]
        if isinstance(step, DataGenerationStep):
            step.saver = self._generated_data_saver
        elif isinstance(step, ExperimentExecutionStep):
            step.loader = self._generated_data_loader
            step.saver = self._result_saver
        elif isinstance(step, ReportGenerationStep):
            step.loader = self._result_loader
        else:
            raise ValueError(
                f"Unexpected type of {step}."
                f" Must be one of DataGenerationStep, ExperimentExecutionStep or ReportGenerationStep"
            )
        cpd_logger.debug(f"{step} Storages: FINISH SETUP")

    @log_exceptions
    def config_pipeline(self) -> None:
        """Configure the pipeline by:

        1. Verifying step compatibility
        2. Initializing storage handlers
        3. Setting up step storage connections

        :raises ValueError: If storage handlers are not properly initialized
        """
        for step_index in range(len(self.steps) - 1):
            step_1, step_2 = self.steps[step_index], self.steps[step_index + 1]
            self._check_two_steps(step_1, step_2)
        cpd_logger.debug("The compatibility of the steps has been verified")

        # TODO we have all data to create storages, so we create fields:
        #  _generated_data_saver, _generated_data_loader, _result_saver, _result_loader

        if not (self._generated_data_saver or self._generated_data_loader):
            missed_field = "generated_data_saver" if not self._generated_data_saver else "generated_data_loader"
            raise ValueError(
                f"An error occurred during Database initialization for the generated data"
                f" ({missed_field} was not initialized)"
            )

        if not (self._result_saver or self._result_loader):
            missed_field = "result_saver" if not self._result_saver else "result_storage_fields"
            raise ValueError(
                f"An error occurred during Database initialization for the result ({missed_field} was not initialized)"
            )

        cpd_logger.debug("Storages initialized")

        for step in self.steps:
            self._setup_step_storage(step)

        cpd_logger.debug("Saver and loader are set for each of the steps")
        cpd_logger.info("The pipeline has been successfully configured")
        cpd_logger.debug(f"Gen Data Storage: {self._generated_data_storage_fields}")
        cpd_logger.debug(f"Result Storage: {self._result_storage_fields}")

    def run(self) -> None:
        """
        Execute all steps in the pipeline sequentially.

        Each step's output metadata is accumulated and passed to subsequent steps.

        :note: The pipeline must be properly configured before running with 'config_pipeline' method
        """

        for step in self.steps:
            cpd_logger.info(f"{step}: START")
            step_result = step(**self._meta_data)
            self._meta_data = self._meta_data | step_result
            cpd_logger.info(f"{step}: FINISH")
        cpd_logger.info("Pipeline finished")
