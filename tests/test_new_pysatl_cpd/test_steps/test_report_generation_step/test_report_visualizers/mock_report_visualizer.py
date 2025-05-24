from pathlib import Path
from typing import Optional

from new_pysatl_cpd.steps.report_generation_step.report_visualizers.report_visualizer import ReportVisualizer


class MockReportVisualizer(ReportVisualizer):
    """Mock implementation of ReportVisualizer for testing purposes.

    This class provides a simple implementation that simulates visualization
    generation for testing the report generation pipeline.
    """

    def __init__(
        self,
        path_to_save: Path = Path("new_pysatl_cpd/results"),
        file_name: str = "report",
        builder_result_fields: Optional[set[str] | dict[str, str]] = None,
    ):
        """Initialize the mock report visualizer.

        Args:
            path_to_save: Directory path for saving visualizations
            file_name: Base filename for output files
            builder_result_fields: Fields to process in visualizations
        """
        super().__init__(path_to_save, file_name, builder_result_fields)
        self._visualization_count = 0

    def _draw(self, report_builder_result: dict[str, float]) -> Optional[dict[str, float]]:
        """Simulate visualization generation.

        Args:
            report_builder_result: Filtered report data for visualization

        Returns:
            Optional dictionary containing visualization metadata
        """
        self._visualization_count += 1

        return {
            "visualization_count": float(self._visualization_count),
            "metrics_visualized": float(len(report_builder_result)),
            "visualization_timestamp": float(self._visualization_count * 1000)
        }
