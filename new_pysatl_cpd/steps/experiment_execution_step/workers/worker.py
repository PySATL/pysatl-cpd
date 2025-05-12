from abc import abstractmethod
from collections.abc import Iterable
from typing import Any

from new_pysatl_cpd.steps.step_processor import StepProcessor


class Worker(StepProcessor):
    @abstractmethod
    def run(self, **kwargs: Any) -> Iterable[dict[str, float]]: ...
