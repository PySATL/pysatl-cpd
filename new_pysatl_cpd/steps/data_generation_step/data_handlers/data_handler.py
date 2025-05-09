from abc import ABC, abstractmethod
from typing import Iterator

from new_pysatl_cpd.steps.step_processor import StepProcessor


class DataHandler(ABC, StepProcessor):
    @abstractmethod
    def get_data(self, **kwargs) -> Iterator[dict[str, float]]:
        ...
