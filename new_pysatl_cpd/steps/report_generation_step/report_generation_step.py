from pathlib import Path
from typing import Any, Optional

from new_pysatl_cpd.steps.report_generation_step.reporters.reporter import Reporter
from new_pysatl_cpd.steps.step import Step
from new_pysatl_cpd.storages.loaders.default_loader import DefaultLoader
from new_pysatl_cpd.storages.loaders.loader import Loader


class ReportGenerationStep(Step):
    def __init__(
        self,
        reporter: Reporter,
        loader: Optional[Loader] = None,
        name: str = "Step",
        input_storage_names: Optional[set[str]] = None,
        output_storage_names: Optional[set[str] | dict[str, str]] = None,
        input_step_names: Optional[set[str] | dict[str, str]] = None,
        output_step_names: Optional[set[str] | dict[str, str]] = None,
        config: Optional[Path] = None,
    ):
        super().__init__(
            name,
            input_storage_names,
            output_storage_names,
            input_step_names,
            output_step_names,
            config,
        )
        self._reporter = reporter
        self._loader = loader if loader else DefaultLoader()
        self._available_next_classes = [ReportGenerationStep]

    def __call__(self, **kwargs: Any) -> dict[str, float]:
        # TODO: load data
        storage_input: dict[str, float] = dict()
        renamed_storage_input = self._get_storage_input(storage_input)
        renamed_step_input = self._get_step_input(kwargs)

        report_result = self._reporter.create_report(**renamed_storage_input, **renamed_step_input)
        renamed_step_output = self._get_step_output(report_result) if report_result else dict()

        return renamed_step_output
