from abc import ABC, abstractmethod
from typing import Any


class ReportBuilder(ABC):
    @abstractmethod
    def build(self, **kwargs: Any) -> None: ...
