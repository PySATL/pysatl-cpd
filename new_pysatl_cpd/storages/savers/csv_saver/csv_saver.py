import csv
from pathlib import Path

from new_pysatl_cpd.storages.savers.saver import Saver


class SaverCSV(Saver):
    """CSV data saver"""

    def __init__(self, step_storages_name: str = "generation"):
        self.directory = Path("experiment_storages") / "csv" / step_storages_name
        self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, storage_name: str, data: dict[str, float]) -> None:
        """Saves data to experiment_storage/[step_storage_name]/[storage_name].csv with key,value from given dict"""
        filename = self.directory / f"{storage_name}.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])

            for key, value in data.items():
                writer.writerow([key, value])
