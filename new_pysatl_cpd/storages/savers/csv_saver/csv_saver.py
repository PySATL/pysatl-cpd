import csv
from pathlib import Path

from new_pysatl_cpd.storages.savers.saver import Saver


class SaverCSV(Saver):
    """Saves data to a CSV file, appending new rows with timestamps."""

    def __init__(self, step_storages_name: str = "generation"):
        self.directory = Path("experiment_storage/" + step_storages_name)
        self.directory.mkdir(parents=True, exist_ok=True)

    # def _ensure_file_exists(self):
    #     """Create file with headers if it doesn't exist."""
    #     if not Path(self.filename).exists():
    #         with open(self.filename, "w", newline="") as f:
    #             writer = csv.writer(f)
    #             writer.writerow(["key", "value"])

    def __call__(self, storage_name: str, data: dict[str, float]) -> None:
        filename = self.directory / f"{storage_name}.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])  # Заголовки

            for key, value in data.items():
                writer.writerow([key, value])
