from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from new_pysatl_cpd.storages.loaders.loader import Loader
from new_pysatl_cpd.storages.savers.saver import Saver


class Step(ABC):
    """Abstract base class representing a processing step in a pipeline.

    This class defines the core interface and common functionality for all pipeline steps,
    including data validation, input/output processing, and step chaining.

    :param name: Human-readable name for the step (default: "Step")
    :param input_storage_names: Required input storage fields (set or dict for renaming)
    :param output_storage_names: Output storage fields (set or dict for renaming)
    :param input_step_names: Required input fields from previous steps (set or dict for renaming)
    :param output_step_names: Output fields for next steps (set or dict for renaming)
    :param config: Path to configuration file

    :ivar name: Step identifier
    :ivar input_storage_names: Required input storage fields
    :ivar output_storage_names: Output storage fields
    :ivar input_step_names: Required input fields from previous steps
    :ivar output_step_names: Output fields for next steps
    :ivar _config: Step configuration path
    :ivar _next: Next step in pipeline
    :ivar _available_next_classes: Allowed types for next steps
    :ivar _saver: Data saver instance
    :ivar _loader: Data loader instance

    .. rubric:: Key Functionality

    - Input/output data validation and transformation
    - Step chaining with type checking
    - Automatic field renaming when dictionaries are provided
    - Abstract method enforcement for concrete implementations

    .. rubric:: Implementation Requirements

    Subclasses must:

    1. Implement :meth:`_validate_storages` to verify storage connections
    2. Implement :meth:`process` (via ABC inheritance)
    3. Define ``_available_next_classes`` to control valid step sequences
    """

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
        self._saver: Optional[Saver] = None
        self._loader: Optional[Loader] = None

    def set_next(self, next_step: "Step") -> None:
        """Set the next step in the pipeline with type validation.

        :param next_step: The subsequent step to execute
        :raises ValueError: If next_step type isn't in _available_next_classes
        """
        available_next_classes = self._available_next_classes

        if type(next_step) not in available_next_classes:
            raise ValueError(
                f"Only {available_next_classes} available for {self.name} ({type(self)})."
                f" But {next_step.name} ({type(next_step)}) was given"
            )

    def _filter_and_rename(
        self, source_dict: dict[str, float], reference_dict: dict[str, str] | set[str]
    ) -> dict[str, float]:
        """Filter and optionally rename dictionary fields based on reference.

        :param source_dict: Input data dictionary
        :param reference_dict: Field names (set) or mapping (dict)
        :return: Filtered/renamed dictionary
        :raises ValueError: If required fields are missing
        """

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
        """Input from storage for Step after renaming

        :param input_data: global input from storage
        """
        return self._filter_and_rename(input_data, self.input_storage_names)

    def _get_step_input(self, input_data: dict[str, float]) -> dict[str, float]:
        """Input from previous step for current step after renaming

        :param input_data: global input from previous step
        """
        return self._filter_and_rename(input_data, self.input_step_names)

    def _get_step_output(self, output_data: dict[str, float]) -> dict[str, float]:
        """Output for storage from Step after renaming

        :param output_data: global output from Step
        """
        return self._filter_and_rename(output_data, self.output_step_names)

    def _get_storage_output(self, output_data: dict[str, float]) -> dict[str, float]:
        """Output for next step from current step after renaming

        :param output_data: global output from current step
        """
        return self._filter_and_rename(output_data, self.output_storage_names)

    @abstractmethod
    def _validate_storages(self) -> bool:
        """Validate that required storage connections are established.

        :return: True if all storage dependencies are satisfied
        """
        ...

    @abstractmethod
    def process(self, **kwargs: Any) -> dict[str, float]:
        """Execute the core processing logic of the step (must be implemented by subclasses).

        This is the main method where the step's functionality should be implemented.
        It receives input data and returns processed results.

        :param kwargs: Arbitrary keyword arguments containing input data for processing.
        :return: Dictionary containing processed results where:
                 - Keys are output field names (strings)
                 - Values are numeric results (floats)

        .. note::
            - All required inputs should be declared in the step's ``input_storage_names``
              and ``input_step_names`` during initialization
            - All outputs should be declared in ``output_storage_names`` and ``output_step_names``

        .. rubric:: Typical Implementation Pattern

        Example::

            def process(self, **kwargs) -> dict[str, float]:
                input_data = self._get_storage_input(kwargs)
                metadata = self._get_step_input(kwargs)

                # Processing logic here
                result = perform_calculation(input_data, metadata)

                return {
                    "output_value": result,
                    "status": 1.0,  # success flag
                }
        """
        ...

    def __call__(self, **kwargs: Any) -> dict[str, float]:
        """Execute the step with given inputs.

        :param kwargs: Input data for processing
        :return: Processed output data
        :raises ValueError: If storages aren't properly configured
        """
        if not self._validate_storages():
            raise ValueError(f"{self} ran without seting up DataBase. (Try to use this step in Pipeline)")
        result = self.process(**kwargs)
        return result

    def __str__(self) -> str:
        return f"{self.name} ({type(self).__name__})"

    @property
    def saver(self) -> Optional[Saver]:
        return self._saver

    @saver.setter
    def saver(self, saver: Saver) -> None:
        self._saver = saver

    @property
    def loader(self) -> Optional[Loader]:
        return self._loader

    @loader.setter
    def loader(self, loader: Loader) -> None:
        self._loader = loader
