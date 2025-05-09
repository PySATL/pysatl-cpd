from new_pysatl_cpd.storages.savers.saver import Saver


class DefaultSaver(Saver):
    def __call__(self, data: dict[str, float]) -> None:
        pass
