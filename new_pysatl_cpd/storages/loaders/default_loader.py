from new_pysatl_cpd.storages.loaders.loader import Loader


class DefaultLoader(Loader):
    def __call__(self, data: dict[str, float]) -> None:
        pass
