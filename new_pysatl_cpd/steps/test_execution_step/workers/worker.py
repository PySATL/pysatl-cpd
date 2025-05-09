from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from new_pysatl_cpd.steps.step_processor import StepProcessor


class Worker(ABC, StepProcessor):
    @abstractmethod
    def run(self, **kwargs: Any) -> Iterator[dict[str, float]]: ...
