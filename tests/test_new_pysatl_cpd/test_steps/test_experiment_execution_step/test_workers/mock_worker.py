from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from new_pysatl_cpd.steps.experiment_execution_step.workers.worker import Worker


class MockWorker(Worker):
    """Mock implementation of Worker for testing purposes.

    This class provides a simple implementation that simulates work execution
    and yields predictable results for testing the ExperimentExecutionStep
    and related components.
    """

    def __init__(self, name: str = "Worker",
                 input_storage_names: Optional[set[str]] = None,
                 output_storage_names: Optional[set[str]] = None,
                 input_step_names: Optional[set[str]] = None,
                 output_step_names: Optional[set[str]] = None,
                 previous_step_data: Optional[dict[str, Any]] = None,
                 config: Optional[Path] = None,
                 num_iterations: int = 5,
                 work_per_iteration: int = 3):
        """Initialize the mock worker.

        Args:
            name: Name of the worker
            input_storage_names: Set of input storage names
            output_storage_names: Set of output storage names
            input_step_names: Set of input step names
            output_step_names: Set of output step names
            previous_step_data: Data from previous steps
            config: Path to configuration file
            num_iterations: Number of work iterations to simulate
            work_per_iteration: Number of metrics to generate per iteration
        """
        super().__init__(name, input_storage_names, output_storage_names,
                        input_step_names, output_step_names,
                        previous_step_data, config)
        self.num_iterations = num_iterations
        self.work_per_iteration = work_per_iteration

    def run(self, *args, **kwargs: Any) -> Iterable[dict[str, float]]:
        """Simulate work execution and yield mock results.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Yields:
            Dictionary containing mock execution results for each iteration
        """
        for iteration in range(self.num_iterations):
            # Simulate some work by generating mock metrics
            results = {
                f"metric_{i}": float(iteration * 100 + i)
                for i in range(self.work_per_iteration)
            }
            # Add a progress metric
            results["progress"] = float(iteration + 1) / self.num_iterations
            yield results
