"""
Module, that contains basic implementation of ReportVisualizer.

ReportVisualizer, that prints located change points into txt file. Used in example.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path
from typing import Optional

from new_pysatl_cpd.custom_types import StorageValues
from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.steps.report_generation_step.report_visualizers.report_visualizer import ReportVisualizer


class CpTextVisualizer(ReportVisualizer):
    """Dummy Report Visualizer without realisation"""

    def __init__(
        self,
        path_to_save: Path = Path("new_pysatl_cpd/results"),
        file_name: str = "report",
        builder_result_fields: Optional[set[str] | dict[str, str]] = None,
    ):
        builder_result_fields = builder_result_fields if builder_result_fields else {"change_points"}
        super().__init__(path_to_save, file_name, builder_result_fields)

    def _draw(self, report_builder_result: dict[str, StorageValues]) -> Optional[dict[str, StorageValues]]:
        cpd_logger.debug("CpTextVisualizer draw method start")
        path = self._path_to_save
        file_name = self._file_name
        cpd_logger.info(report_builder_result)
        with open(f"{path}/{file_name}.txt", "w", encoding="utf-8") as file:
            file.write(f"Located change points: {report_builder_result['change_points']}\n")
        return None
