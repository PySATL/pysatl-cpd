import csv

from new_pysatl_cpd.storages.loaders.loader import Loader


class LoaderCSV(Loader):
    """Loads the most recent values for requested keys from CSV file."""

    def __init__(self, filename: str = "data.csv"):
        self.filename = filename

    def __call__(self, data_keys: set[str]) -> dict[str, float]:
        try:
            with open(self.filename) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except FileNotFoundError:
            return {key: 0.0 for key in data_keys}

        result = {}
        for key in data_keys:
            key_rows = [row for row in rows if row["key"] == key]
            if key_rows:
                last_row = key_rows[-1]
                result[key] = float(last_row["value"])
            else:
                result[key] = 0.0

        return result
