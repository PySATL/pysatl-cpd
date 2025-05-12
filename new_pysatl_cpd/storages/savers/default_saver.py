from new_pysatl_cpd.storages.savers.saver import Saver


class DefaultSaver(Saver):
    """Dummy saver without realisation"""

    def __call__(self, data: dict[str, float]) -> None:
        pass
