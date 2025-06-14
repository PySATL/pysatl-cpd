"""
Module contains dummy implementation of a report visualizer for testing purposes.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Optional

from new_pysatl_cpd.custom_types import StorageValues
from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.steps.report_generation_step.report_visualizers.report_visualizer import ReportVisualizer


class DummyReportVisualizer(ReportVisualizer):
    """Dummy Report Visualizer without realisation"""

    def _draw(self, report_builder_result: dict[str, StorageValues]) -> Optional[dict[str, StorageValues]]:
        cpd_logger.debug("DummyReportVisualizer draw method")
        path = self._path_to_save
        file_name = self._file_name
        with open(f"{path}/{file_name}.txt", "w", encoding="utf-8") as file:
            file.write(f"a={report_builder_result['a']}\n")
            file.write(f"b={report_builder_result['b']}\n")
            file.write(f"c={report_builder_result['c']}\n")
            file.write(f"sum={report_builder_result['s']}\n")
        return None
