import csv
from pathlib import Path

from new_pysatl_cpd.storages.loaders.loader import Loader


class LoaderCSV(Loader):
    """CSV data loader"""

    def __init__(self, step_storages_name: str = "generation"):
        self.directory = Path("experiment_storages/csv/" + step_storages_name)

    def __call__(self, data_keys: set[str]) -> dict[str, dict[int, float]]:
        """Loads data from files, matching experiment_storage/[step_storage_name]/[data_keys].csv pattern"""
        result = {}
        for key in data_keys:
            filename = self.directory / f"{key}.csv"
            file_data = {}
            if filename.exists():
                with open(filename) as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row_id, row in enumerate(reader, start=1):
                        min_columns_counter = 2
                        if len(row) >= min_columns_counter:
                            try:
                                file_data[row_id] = float(row[1])
                            except (ValueError, IndexError):
                                continue
            result[key] = file_data

        return result
