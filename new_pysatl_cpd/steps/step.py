from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class Step(ABC):
    def __init__(
        self,
        name: str = "Step",
        input_storage_names: Optional[set[str]] = None,
        output_storage_names: Optional[set[str] | dict[str, str]] = None,
        input_step_names: Optional[set[str] | dict[str, str]] = None,
        output_step_names: Optional[set[str] | dict[str, str]] = None,
        config: Optional[Path] = None,
    ):
        self.name = name
        self.input_storage_names = input_storage_names if input_storage_names else set()
        self.output_storage_names = output_storage_names if output_storage_names else set()
        self.input_step_names = input_step_names if input_step_names else set()
        self.output_step_names = output_step_names if output_step_names else set()
        self._config = config
        self._next: Optional[Step] = None
        self._available_next_classes: list[type[Step]] = []

    def set_next(self, next_step: "Step") -> None:
        available_next_classes = self._available_next_classes

        if type(next_step) not in available_next_classes:
            raise ValueError(
                f"Only {available_next_classes} available for {self.name} ({type(self)})."
                f" But {next_step.name} ({type(next_step)}) was given"
            )

    def _filter_and_rename(
        self, source_dict: dict[str, float], reference_dict: dict[str, str] | set[str]
    ) -> dict[str, float]:
        result = dict()
        for key in reference_dict:
            if key not in source_dict:
                raise ValueError(f"No {key} in input step data for Step: {self.name}")

            # renaming keys
            if isinstance(reference_dict, dict):
                result[reference_dict[key]] = source_dict[key]
            else:
                result[key] = source_dict[key]

        return result

    def _get_storage_input(self, input_data: dict[str, float]) -> dict[str, float]:
        return self._filter_and_rename(input_data, self.input_storage_names)

    def _get_step_input(self, input_data: dict[str, float]) -> dict[str, float]:
        return self._filter_and_rename(input_data, self.input_step_names)

    def _get_step_output(self, output_data: dict[str, float]) -> dict[str, float]:
        return self._filter_and_rename(output_data, self.output_step_names)

    def _get_storage_output(self, output_data: dict[str, float]) -> dict[str, float]:
        return self._filter_and_rename(output_data, self.output_storage_names)

    @abstractmethod
    def __call__(self, **kwargs: Any) -> dict[str, float]: ...
