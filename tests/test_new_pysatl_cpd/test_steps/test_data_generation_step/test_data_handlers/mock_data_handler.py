from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from new_pysatl_cpd.steps.data_generation_step.data_handlers.data_handler import DataHandler


class MockDataHandler(DataHandler):
    """Mock implementation of DataHandler for testing purposes.

    This class provides a simple implementation that generates predictable mock data
    for testing the DataGenerationStep and related components.
    """

    def __init__(self, name: str = "Step",
                 input_storage_names: Optional[set[str]] = None,
                 output_storage_names: Optional[set[str]] = None,
                 input_step_names: Optional[set[str]] = None,
                 output_step_names: Optional[set[str]] = None,
                 previous_step_data: Optional[dict[str, Any]] = None,
                 config: Optional[Path] = None,
                 num_chunks: int = 5,
                 values_per_chunk: int = 3):
        """Initialize the mock data handler.

        Args:
            num_chunks: Number of data chunks to generate
            values_per_chunk: Number of values in each chunk
        """
        super().__init__(name, input_storage_names, output_storage_names, input_step_names, output_step_names,
                         previous_step_data, config)
        self.num_chunks = num_chunks
        self.values_per_chunk = values_per_chunk

    def get_data(self, *args, **kwargs: Any) -> Iterable[dict[str, float]]:
        """Generate mock data chunks for testing.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Yields:
            Dictionary containing mock data values for each chunk
        """
        for chunk_idx in range(self.num_chunks):
            mock_data = {
                f"value_{i}": float(chunk_idx * 10 + i)
                for i in range(self.values_per_chunk)
            }
            yield mock_data
