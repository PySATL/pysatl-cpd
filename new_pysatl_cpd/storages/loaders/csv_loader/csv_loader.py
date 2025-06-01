import csv
from pathlib import Path

from new_pysatl_cpd.storages.loaders.loader import Loader


class LoaderCSV(Loader):
    """Loads the most recent values for requested keys from CSV file."""

    def __init__(self, step_storages_name: str = "generation"):
        self.directory = Path("experiment_storage/" + step_storages_name)

    def __call__(self, data_keys: set[str]) -> dict[str, dict[int, float]]:
        result = {}

        for key in data_keys:
            filename = self.directory / f"{key}.csv"
            file_data = {}

            if filename.exists():
                with open(filename) as f:
                    reader = csv.reader(f)
                    next(reader)  # Пропускаем заголовок (key,value)

                    for row_id, row in enumerate(reader, start=1):
                        TWO = 2
                        if len(row) >= TWO:  # Проверяем, что есть и ключ и значение
                            try:
                                file_data[row_id] = float(row[1])
                            except (ValueError, IndexError):
                                continue

            result[key] = file_data

        return result
