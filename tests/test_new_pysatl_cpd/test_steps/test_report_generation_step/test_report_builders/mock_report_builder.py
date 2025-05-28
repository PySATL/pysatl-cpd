from typing import Any, Optional

from new_pysatl_cpd.steps.report_generation_step.report_builders.report_builder import ReportBuilder


class MockReportBuilder(ReportBuilder):
    """Mock implementation of ReportBuilder for testing purposes.

    This class provides a simple implementation that generates predictable mock report data
    for testing the report generation pipeline.
    """

    def __init__(self, builder_result_fields: Optional[set[str] | dict[str, str]] = None):
        """Initialize the mock report builder.

        Args:
            builder_result_fields: Fields to include in final output. Can be:
                - A set of field names to keep (no renaming)
                - A dict mapping {source_name: target_name} for renaming
        """
        super().__init__(builder_result_fields)

    def _build(self, *args, **kwargs: Any) -> dict[str, float]:
        """Generate mock report data.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            Dictionary containing mock report metrics
        """
        return {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
            "f1_score": 0.90,
            "loss": 0.15,
            "training_time": 120.5,
            "inference_time": 0.05,
        }
