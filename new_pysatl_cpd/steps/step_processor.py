from abc import ABC
from pathlib import Path
from typing import Any, Optional


class StepProcessor(ABC):
    """Abstract base class defining the interface for step processor (DataHandler for DataGenerationStep,
    Worker for TestExecutionStep, Reporter for ReportGenerationStep).

       :param name: Human-readable identifier for this step (default: "Step")
       :param input_storage_names: Set of required input data fields from storage
       :param output_storage_names: Set of output data fields this step will produce
       :param input_step_names: Set of required metadata fields from previous steps
       :param output_step_names: Set of metadata fields this step will produce
       :param previous_step_data: Dictionary containing outputs from preceding steps
       :param config: Path to configuration file for this step

       :ivar name: Identifier for this processing step
       :ivar input_storage_names: Required input data fields from storage
       :ivar output_storage_names: Output data fields this step produces
       :ivar input_step_names: Required metadata fields from previous steps
       :ivar output_step_names: Metadata fields this step produces
       :ivar previous_step_data: Data dictionary from preceding steps
       :ivar config: Path to step's configuration file

    """

    def __init__(
        self,
        name: str = "Step",
        input_storage_names: Optional[set[str]] = None,
        output_storage_names: Optional[set[str]] = None,
        input_step_names: Optional[set[str]] = None,
        output_step_names: Optional[set[str]] = None,
        previous_step_data: Optional[dict[str, Any]] = None,
        config: Optional[Path] = None,
    ) -> None:
        self.name = name
        self.input_storage_names = input_storage_names if input_storage_names else set()
        self.output_storage_names = output_storage_names if output_storage_names else set()
        self.input_step_names = input_step_names if input_step_names else set()
        self.output_step_names = output_step_names if output_step_names else set()
        self.previous_step_data = previous_step_data if previous_step_data else dict()
        self.config = config
