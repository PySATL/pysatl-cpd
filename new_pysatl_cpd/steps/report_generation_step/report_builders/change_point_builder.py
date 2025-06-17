""" """

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, Optional

from new_pysatl_cpd.custom_types import StorageValues
from new_pysatl_cpd.steps.report_generation_step.report_builders.report_builder import ReportBuilder


class CpBuilder(ReportBuilder):
    """Builder for getting change points from result"""

    def __init__(
        self,
        builder_result_fields: Optional[set[str] | dict[str, str]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        builder_result_fields = builder_result_fields if builder_result_fields else {"change_points"}
        super().__init__(builder_result_fields)

    def _build(self, change_points: list[int], *args: Any, **kwargs: Any) -> dict[str, StorageValues]:
        return {"change_points": change_points}
