from typing import Optional

from new_pysatl_cpd.logger import log_exceptions, logger
from new_pysatl_cpd.steps.data_generation_step.data_generation_step import DataGenerationStep
from new_pysatl_cpd.steps.report_generation_step.report_generation_step import ReportGenerationStep
from new_pysatl_cpd.steps.step import Step
from new_pysatl_cpd.steps.test_execution_step.test_execution_step import TestExecutionStep
from new_pysatl_cpd.storages.loaders.loader import Loader
from new_pysatl_cpd.storages.savers.saver import Saver


# TODO Storages
class Pipeline:
    def __init__(self, steps: list[Step]):
        self.steps = steps
        self._generated_data_storage_fields: set[str] = set()
        self._result_storage_fields: set[str] = set()
        self._meta_data: dict[str, float] = dict()
        self._generated_data_saver: Optional[Saver] = None
        self._generated_data_loader: Optional[Loader] = None
        self._result_saver: Optional[Saver] = None
        self._result_loader: Optional[Loader] = None

    def _check_two_steps(self, step_1: Step, step_2: Step) -> None:
        if isinstance(step_1, DataGenerationStep):
            storage_fields = self._generated_data_storage_fields
        elif isinstance(step_1, (TestExecutionStep, ReportGenerationStep)):
            storage_fields = self._result_storage_fields
        else:
            raise ValueError(
                f"{step_1} is unexpected Step (not one of DataGenerationStep, TestExecutionStep, ReportGenerationStep)"
            )

        if isinstance(step_1.output_storage_names, set):
            storage_fields = storage_fields.union(step_1.output_storage_names)
        else:
            storage_fields = storage_fields.union(step_1.output_storage_names.values())

        for key in step_1.output_step_names:
            self._meta_data[key] = 0

        if not step_2.input_storage_names.issubset(storage_fields):
            missed_fields = step_2.input_storage_names - storage_fields
            raise KeyError(
                f" For {step_2} to work, there must be values {missed_fields} in the storage."
                f" Check if this fields are created accurately in the previous steps."
                f" (If these fields are not needed for {step_2} to work,"
                f" then remove them from the input_storage_names field)"
            )

        input_step_names = (
            step_2.input_step_names
            if isinstance(step_2.input_step_names, set)
            else set(step_2.input_step_names.values())
        )

        if not input_step_names.issubset(self._meta_data.keys()):
            missed_fields = input_step_names - self._meta_data.keys()
            raise KeyError(
                f" For {step_2} to work, there must be values {missed_fields} returned from previous steps"
                f" as meta data."
                f" Check if this fields are created accurately in the previous steps."
                f" (If these fields are not needed for {step_2} to work,"
                f" then remove them from the input_step_names field)"
            )

    def _setup_step_storage(self, step: Step) -> None:
        if isinstance(step, DataGenerationStep):
            step.saver = self._generated_data_saver
        elif isinstance(step, TestExecutionStep):
            step.loader = self._generated_data_loader
            step.saver = self._result_saver
        elif isinstance(step, ReportGenerationStep):
            step.loader = self._result_loader
        else:
            raise ValueError(
                f"Unexpected type of {step}."
                f" Must be one of DataGenerationStep, TestExecutionStep or ReportGenerationStep"
            )

    @log_exceptions
    def config_pipeline(self) -> None:
        for step_index in range(len(self.steps) - 1):
            step_1, step_2 = self.steps[step_index], self.steps[step_index + 1]
            self._check_two_steps(step_1, step_2)
        logger.debug("The compatibility of the steps has been verified")

        # TODO we have all data to create storages, so we create fields:
        #  _generated_data_saver, _generated_data_loader, _result_saver, _result_loader

        if not (self._generated_data_saver or self._generated_data_loader):
            missed_field = "generated_data_saver" if not self._generated_data_saver else "generated_data_loader"
            raise ValueError(
                f"An error occurred during Database initialization for the generated data"
                f" ({missed_field} was not initialized)"
            )

        if not (self._result_saver or self._result_storage_fields):
            missed_field = "result_saver" if not self._result_saver else "result_storage_fields"
            raise ValueError(
                f"An error occurred during Database initialization for the result ({missed_field} was not initialized)"
            )

        logger.debug("Storages initialized")

        for step in self.steps:
            self._setup_step_storage(step)

        logger.debug("Saver and loader are set for each of the steps")
        logger.info("The pipeline has been successfully configured")

    def run(self) -> None:
        for step in self.steps:
            logger.info(f"{step}: START")
            step_result = step(**self._meta_data)
            self._meta_data = self._meta_data | step_result
            logger.info(f"{step}: OK")
        logger.info("Pipeline finished")
