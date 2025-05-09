from pathlib import Path
from typing import Optional, Any


class StepProcessor:
    def __init__(self, name: str = "Step",
                 input_storage_names: Optional[set[str]] = None,
                 output_storage_names: Optional[set[str]] = None,
                 input_step_names: Optional[set[str]] = None,
                 output_step_names: Optional[set[str]] = None,
                 previous_step_data: Optional[dict[str, Any]] = None,
                 config: Optional[Path] = None) -> None:
        self.name = name
        self.input_storage_names = input_storage_names if input_storage_names else set()
        self.output_storage_names = output_storage_names if output_storage_names else set()
        self.input_step_names = input_step_names if input_step_names else set()
        self.output_step_names = output_step_names if output_step_names else set()
        self.previous_step_data = previous_step_data if previous_step_data else dict()
        self.config = config
