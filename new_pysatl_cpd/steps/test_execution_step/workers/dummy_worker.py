from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.steps.test_execution_step.workers.worker import Worker


class DummyWorker(Worker):
    def __init__(
        self,
        data_to_return: Optional[dict[str, float]] = None,
        name: str = "Step",
        input_storage_names: Optional[set[str]] = None,
        output_storage_names: Optional[set[str]] = None,
        input_step_names: Optional[set[str]] = None,
        output_step_names: Optional[set[str]] = None,
        previous_step_data: Optional[dict[str, Any]] = None,
        config: Optional[Path] = None,
    ) -> None:
        super().__init__(
            name,
            input_storage_names,
            output_storage_names,
            input_step_names,
            output_step_names,
            previous_step_data,
            config,
        )
        self._data_to_return = data_to_return if data_to_return else dict()

    def run(self, **kwargs: Any) -> Iterable[dict[str, float]]:
        cpd_logger.debug("DummyWorker run method")
        return [self._data_to_return]
