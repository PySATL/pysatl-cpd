from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any

from new_pysatl_cpd.steps.report_generation_step.report_builders.report_builder import ReportBuilder
from new_pysatl_cpd.steps.report_generation_step.report_visualizers.report_visualizer import ReportVisualizer
from new_pysatl_cpd.steps.step_processor import StepProcessor


class Reporter(ABC, StepProcessor):
    def __init__(self,
                 report_builder: ReportBuilder,
                 report_visualizer: ReportVisualizer,
                 path_to_save: Path,
                 name: str = "Step",
                 input_storage_names: Optional[set[str]] = None,
                 output_storage_names: Optional[set[str]] = None,
                 input_step_names: Optional[set[str]] = None,
                 output_step_names: Optional[set[str]] = None,
                 previous_step_data: Optional[dict[str, Any]] = None,
                 config: Optional[Path] = None) -> None:
        super().__init__(name, input_storage_names, output_storage_names, input_step_names, output_step_names,
                         previous_step_data, config)
        self._report_builder = report_builder
        self._report_visualizer = report_visualizer
        self._path_to_save = path_to_save

    @abstractmethod
    def create_report(self, **kwargs) -> Optional[dict[str, float]]:
        ...
