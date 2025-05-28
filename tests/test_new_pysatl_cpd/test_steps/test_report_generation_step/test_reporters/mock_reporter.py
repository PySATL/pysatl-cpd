from pathlib import Path
from typing import Any, Optional

from new_pysatl_cpd.steps.report_generation_step.report_builders.report_builder import ReportBuilder
from new_pysatl_cpd.steps.report_generation_step.report_visualizers.report_visualizer import ReportVisualizer
from new_pysatl_cpd.steps.report_generation_step.reporters.reporter import Reporter


class MockReporter(Reporter):
    """Mock implementation of Reporter for testing purposes.

    This class provides a simple implementation that coordinates mock report building
    and visualization for testing the report generation pipeline.
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
        result: Optional[dict[str, float]] = None,
    ) -> None:
        """Initialize the mock reporter.

        Args:
            report_builder: Component responsible for generating report data
            report_visualizer: Component responsible for visualizing report data
            name: Human-readable identifier for this step
            input_storage_names: Set of required input data fields from storage
            output_storage_names: Set of output data fields this step will produce
            input_step_names: Set of required metadata fields from previous steps
            output_step_names: Set of metadata fields this step will produce
            previous_step_data: Dictionary containing outputs from preceding steps
            config: Path to configuration file for this step
        """
        super().__init__(
            report_builder=report_builder,
            report_visualizer=report_visualizer,
            name=name,
            input_storage_names=input_storage_names,
            output_storage_names=output_storage_names,
            input_step_names=input_step_names,
            output_step_names=output_step_names,
            previous_step_data=previous_step_data,
            config=config,
        )
        self._report_count = 0
        self.result = result

    def create_report(self, *args, **kwargs: Any) -> Optional[dict[str, float]]:
        """Execute the mock report generation and visualization pipeline.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            Optional dictionary containing report metadata
        """
        self._report_count += 1

        report_builder_result = self._report_builder(*args, **kwargs)

        visualization_result = self._report_visualizer(report_builder_result)

        return (
            {
                "report_count": float(self._report_count),
                "metrics_count": float(len(report_builder_result)),
                "visualization_success": float(1.0 if visualization_result else 0.0),
            }
            if not self.result
            else self.result
        )
