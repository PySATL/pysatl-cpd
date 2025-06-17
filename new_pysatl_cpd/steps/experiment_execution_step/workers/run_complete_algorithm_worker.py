from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import numpy as np

from new_pysatl_cpd.custom_types import StorageValues
from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.steps.experiment_execution_step.workers.worker import Worker
from pysatl_cpd.core.algorithms.abstract_algorithm import Algorithm


class RunCompleteAlgorithmWorker(Worker):
    def __init__(
        self,
        algorithm: Algorithm,
        to_localize: bool = True,
        name: str = "Step",
        input_storage_names: Optional[set[str]] = None,
        output_storage_names: Optional[set[str]] = None,
        input_step_names: Optional[set[str]] = None,
        output_step_names: Optional[set[str]] = None,
        previous_step_data: Optional[dict[str, Any]] = None,
        config: Optional[Path] = None,
    ) -> None:
        input_storage_names = input_storage_names if input_storage_names else {"dataset"}
        output_storage_names = output_storage_names if output_storage_names else {"change_points"}

        super().__init__(
            name,
            input_storage_names,
            output_storage_names,
            input_step_names,
            output_step_names,
            previous_step_data,
            config,
        )
        self.algorithm = algorithm
        self.to_localize = to_localize

    def run(self, dataset: tuple[float]) -> Iterable[dict[str, StorageValues]]:
        # TODO add scrubber
        cpd_logger.info(f"run algo with {dataset}")
        nd_array_dataset = np.array(dataset)
        if self.to_localize:
            result = self.algorithm.localize(nd_array_dataset)
        else:
            result = self.algorithm.detect(nd_array_dataset)
        cpd_logger.info(f"algo result {result}")
        yield {"change_points": result}
