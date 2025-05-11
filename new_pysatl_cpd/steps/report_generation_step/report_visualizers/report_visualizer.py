from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from new_pysatl_cpd.steps.report_generation_step.report_builders.report_builder import ReportBuilder


class ReportVisualizer(ABC):
    def __init__(self, path_to_save: Path = Path("new_pysatl_cpd/results")):
        self.path_to_save = path_to_save

    @abstractmethod
    def draw(self, report_builder: ReportBuilder) -> Optional[dict[str, float]]: ...
