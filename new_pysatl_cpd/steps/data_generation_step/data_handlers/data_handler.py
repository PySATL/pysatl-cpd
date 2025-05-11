from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from new_pysatl_cpd.steps.step_processor import StepProcessor


class DataHandler(ABC, StepProcessor):
    @abstractmethod
    def get_data(self, **kwargs: Any) -> Iterable[dict[str, float]]: ...
