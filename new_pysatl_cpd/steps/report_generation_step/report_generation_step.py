from pathlib import Path
from typing import Any, Optional

from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.steps.report_generation_step.reporters.reporter import Reporter
from new_pysatl_cpd.steps.step import Step


class ReportGenerationStep(Step):
    """Concrete step implementation for generating and visualizing reports in a pipeline.

    This step specializes in report generation by coordinating with a Reporter instance
    to process data and create final reports. It handles the integration between the
    pipeline infrastructure and the reporting components.

    :param reporter: Configured reporter instance that handles actual report generation
    :param name: Human-readable name for this step (default: "Step")
    :param input_storage_names: Required input storage fields (set or dict for renaming)
    :param output_storage_names: Output storage fields (set or dict for renaming)
    :param input_step_names: Required input fields from previous steps (set or dict for renaming)
    :param config: Path to configuration file

    :ivar _reporter: The wrapped reporter instance
    :ivar _available_next_classes: Allowed subsequent step types

    .. rubric:: Execution Workflow

    1. Loads required input data from storage
    2. Processes inputs through the configured Reporter
    3. Handles report generation and visualization
    4. Returns any output metrics
    """

    def __init__(
        self,
        reporter: Reporter,
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
        self._reporter = reporter
        self._available_next_classes = [ReportGenerationStep]
        self._set_storage_data_from_processor(self._reporter)

    def process(self, *args: Any, **kwargs: Any) -> dict[str, dict[Any, Any]]:
        """Execute the report generation process.

        :param kwargs: Input parameters including:
                      - Storage data (accessed via input_storage_names)
                      - Step metadata (accessed via input_step_names)
        :return: Dictionary step Metadata

        .. note::
            - Combines storage data and step metadata for reporting
            - All reporting output handled through Reporter's visualizer
        """

        # TODO: load data

        # REMOVE LATER: DUMMY REALISATION
        # storage_input: dict[str, float] = dict()
        if self.loader is None:
            raise ValueError("Storage loader is not initialized")

        load_from_storage_names = (
            self.input_storage_names
            if isinstance(self.input_storage_names, set)
            else set(self.input_storage_names.keys())
        )

        storage_input: dict[str, dict[Any, Any]] = self.loader(load_from_storage_names)

        renamed_storage_input = self._get_storage_input(storage_input)
        renamed_step_input = self._get_step_input(kwargs)
        cpd_logger.debug(f"Report step storage info: {storage_input}")
        report_result = self._reporter.create_report(**renamed_storage_input, **renamed_step_input)
        renamed_step_output = self._get_step_output(report_result) if report_result else dict()

        return renamed_step_output

    def _validate_storages(self) -> bool:
        """Verify that required storage connections are established.

        :return: True if loader is configured (saver not required)
        """
        return bool(self._loader)
