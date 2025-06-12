"""
Module contains dummy implementation of a reporter for testing purposes.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pathlib import Path
from typing import Any, Optional

from new_pysatl_cpd.steps.report_generation_step.report_builders.report_builder import ReportBuilder
from new_pysatl_cpd.steps.report_generation_step.report_visualizers.report_visualizer import ReportVisualizer
from new_pysatl_cpd.steps.report_generation_step.reporters.reporter import Reporter


class DummyReporter(Reporter):
    def __init__(
        self,
        report_builder: ReportBuilder,
        report_visualizer: ReportVisualizer,
        name: str = "Step",
        previous_step_data: Optional[dict[str, Any]] = None,
        config: Optional[Path] = None,
    ) -> None:
        super().__init__(
            report_builder,
            report_visualizer,
            name,
            {"s"},
            set(),
            set(),
            set(),
            previous_step_data,
            config,
        )
        self._report_builder = report_builder
        self._report_visualizer = report_visualizer
