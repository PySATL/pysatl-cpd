from abc import ABC, abstractmethod
from typing import Iterator

from new_pysatl_cpd.steps.step_processor import StepProcessor


class Worker(ABC, StepProcessor):
    @abstractmethod
    def run(self, *args, **kwargs) -> Iterator[dict[str, float]]:
        ...
