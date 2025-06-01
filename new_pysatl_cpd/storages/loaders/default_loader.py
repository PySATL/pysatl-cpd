from typing import Any

from new_pysatl_cpd.storages.loaders.loader import Loader


class DefaultLoader(Loader):
    """Dummy loader without realisation"""

    def __init__(self, dict_as_db: dict[str, dict[Any, Any]]):
        self.dict_as_db = dict_as_db

    def __call__(self, data_keys: set[str]) -> dict[str, dict[Any, Any]]:
        result = {}
        for key in data_keys:
            result[key] = self.dict_as_db[key]
        return result
