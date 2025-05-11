from typing import Any

from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.steps.report_generation_step.report_builders.report_builder import ReportBuilder


class DummyReportBuilder(ReportBuilder):
    def __init__(self, a: float, b: float, **kwargs: Any):
        self.a = a
        self.b = b

    def build(self, **kwargs: Any) -> None:
        cpd_logger.debug("DummyReportBuilder build method")
