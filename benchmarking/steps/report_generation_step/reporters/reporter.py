from pathlib import Path
from typing import Any, Optional

from benchmarking.custom_types import StorageValues
from benchmarking.logger import cpd_logger
from benchmarking.steps.report_generation_step.report_builders.report_builder import (
    ReportBuilder,
)
from benchmarking.steps.report_generation_step.report_visualizers.report_visualizer import (
    ReportVisualizer,
)
from benchmarking.steps.step_processor import StepProcessor


class Reporter(StepProcessor):
    """Composite step processor that coordinates report generation and visualization.

    Combines a ReportBuilder and ReportVisualizer to create a complete report
    within a processing step. Handles the workflow between data preparation, report
    generation, and visualization.

    :param report_builder: Component responsible for generating report data
    :param report_visualizer: Component responsible for visualizing report data
    :param name: Human-readable name for this step (default: "Step")
    :param input_storage_names: Required input storage fields
    :param output_storage_names: Output storage fields
    :param input_step_names: Required input fields from previous steps
    :param output_step_names: Output fields for next steps
    :param previous_step_data: Data dictionary from preceding steps
    :param config: Path to configuration file (the config is optional. If available, it will be passed to
    the StepProcessor and processed there. Makes it possible to additionally configure a specific StepProcessor)

    :ivar _report_builder: Report generation component
    :ivar _report_visualizer: Report visualization component

    .. rubric:: Workflow

    1. Receives processed data from pipeline
    2. Delegates report generation to ReportBuilder
    3. Passes generated report to ReportVisualizer
    4. Returns None (output handled through visualizer)
    """

    def __init__(
        self,
        report_builder: ReportBuilder,
        report_visualizer: ReportVisualizer,
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
        self._report_builder = report_builder
        self._report_visualizer = report_visualizer

    def create_report(self, *args: Any, **kwargs: Any) -> Optional[dict[str, StorageValues]]:
        """Execute the complete report generation and visualization pipeline.

        :param kwargs: Input data for report generation, typically including:
                      - Processed experiment results
                      - Configuration parameters
                      - Metadata from previous steps
        :return: Metadata from step

        .. note::
            - Coordinates between builder and visualizer components
        """

        report_builder_result = self._report_builder(*args, **kwargs)
        cpd_logger.info(report_builder_result)
        cpd_logger.debug(f"report builder: {report_builder_result}")
        return self._report_visualizer(report_builder_result)
