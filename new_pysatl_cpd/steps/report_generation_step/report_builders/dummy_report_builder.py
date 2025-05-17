from typing import Any, Optional

from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.steps.report_generation_step.report_builders.report_builder import ReportBuilder


class DummyReportBuilder(ReportBuilder):
    """Dummy Report Builder without realisation"""

    def __init__(
        self, a: float, b: float, builder_result_fields: Optional[set[str] | dict[str, str]] = None, **kwargs: Any
    ):
        super().__init__(builder_result_fields)
        self.a = a
        self.b = b

    def _build(self, **kwargs: Any) -> dict[str, float]:
        cpd_logger.debug(f"DummyReportBuilder build method ({kwargs})")

        return {"a": self.a, "b": self.b, "c": self.a + self.b, "s": kwargs["s"]}
