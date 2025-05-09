from pathlib import Path
from typing import Optional

from new_pysatl_cpd.steps.step import Step
from new_pysatl_cpd.steps.test_execution_step.workers.worker import Worker
from new_pysatl_cpd.storages.loaders.loader import Loader
from new_pysatl_cpd.storages.savers.saver import Saver


class TestExecutionStep(Step):
    def __init__(self,
                 worker: Worker,
                 loader: Optional[Loader] = None,
                 saver: Optional[Saver] = None,
                 name: str = "Step",
                 input_storage_names: Optional[set[str]] = None,
                 output_storage_names: Optional[set[str] | dict[str, str]] = None,
                 input_step_names: Optional[set[str] | dict[str, str]] = None,
                 output_step_names: Optional[set[str] | dict[str, str]] = None,
                 config: Optional[Path] = None):
        super().__init__(name, input_storage_names, output_storage_names, input_step_names, output_step_names, config)
        self._worker = worker
        self._loader = loader
        self._saver = saver

    def __call__(self, **kwargs) -> Optional[dict[str, float]]:
        # TODO: load data
        storage_input = dict()
        renamed_storage_input = self._get_storage_input(storage_input)
        renamed_step_input = self._get_step_input(kwargs)

        # Maybe result must be iterable, or remove 'for worker_result ...'
        renamed_step_output = dict()

        for worker_result in self._worker.run(**renamed_storage_input, **renamed_step_input):
            renamed_step_output = self._get_step_output(worker_result)
            renamed_storage_output = self._get_storage_output(worker_result)
            self._saver(renamed_storage_output)

        return renamed_step_output