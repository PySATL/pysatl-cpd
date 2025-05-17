from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.steps.experiment_execution_step.workers.worker import Worker


class DummyWorker(Worker):
    """Dummy Worker without realisation"""

    def __init__(
        self,
        data_to_return: Optional[dict[str, float]] = None,
        name: str = "Step",
        previous_step_data: Optional[dict[str, Any]] = None,
        config: Optional[Path] = None,
    ) -> None:
        super().__init__(
            name,
            {"b"},
            {"s"},
            {"a"},
            set(),
            previous_step_data,
            config,
        )
        self._data_to_return = data_to_return if data_to_return else dict()

    def run(self, a: int, b: int) -> Iterable[dict[str, float]]:
        cpd_logger.debug("DummyWorker run method")
        cpd_logger.debug(f"get for execution: a: {a}, b: {b} ")
        return [{"s": a + b}]
