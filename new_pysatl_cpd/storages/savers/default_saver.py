from new_pysatl_cpd.logger import cpd_logger
from new_pysatl_cpd.storages.savers.saver import Saver


class DefaultSaver(Saver):
    """Dummy saver without realisation"""

    def __init__(self, dict_as_db: dict[str, float]):
        self.dict_as_db = dict_as_db

    def __call__(self, data: dict[str, float]) -> None:
        for key, value in data.items():
            cpd_logger.info(f"Saved: {key}")
            self.dict_as_db[key] = value
