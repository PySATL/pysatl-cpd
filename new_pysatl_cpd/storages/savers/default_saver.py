from typing import Any

from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.storages.savers.saver import Saver


class DefaultSaver(Saver):
    """Dummy saver without realisation"""

    def __init__(self, dict_as_db: dict[str, dict[Any, Any]]):
        self.dict_as_db = dict_as_db

    def __call__(self, storage_name: str, data: dict[str, dict[Any, Any]]) -> None:
        cpd_logger.info(f"Saved: {storage_name}")
        self.dict_as_db[storage_name] = data
