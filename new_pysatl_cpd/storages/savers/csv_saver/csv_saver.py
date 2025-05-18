import csv
from pathlib import Path

from new_pysatl_cpd.storages.savers.saver import Saver


class SaverCSV(Saver):
    """Saves data to a CSV file, appending new rows with timestamps."""

    def __init__(self, filename: str = "data.csv"):
        self.filename = filename
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create file with headers if it doesn't exist."""
        if not Path(self.filename).exists():
            with open(self.filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["key", "value"])

    def __call__(self, data: dict[str, float]) -> None:
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            for key, value in data.items():
                writer.writerow([key, value])
