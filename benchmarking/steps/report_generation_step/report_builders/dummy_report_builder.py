"""
Module contains dummy implementation of a report builder for testing and prototyping report generation.

The DummyReportBuilder provides a minimal ReportBuilder implementation that performs no actual report construction,
serving as a placeholder for pipeline testing and development purposes. It demonstrates basic report
structure with dummy calculations.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, Optional

from benchmarking.custom_types import StorageValues
from benchmarking.logger import cpd_logger
from benchmarking.steps.report_generation_step.report_builders.report_builder import ReportBuilder


class DummyReportBuilder(ReportBuilder):
    """Dummy Report Builder without realisation"""

    def __init__(
        self,
        a: float,
        b: float,
        builder_result_fields: Optional[set[str] | dict[str, str]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(builder_result_fields)
        self.a = a
        self.b = b

    def _build(self, *args: Any, **kwargs: Any) -> dict[str, StorageValues]:
        cpd_logger.debug(f"DummyReportBuilder build method ({kwargs})")

        return {"a": self.a, "b": self.b, "c": self.a + self.b, "s": kwargs["s"]}
