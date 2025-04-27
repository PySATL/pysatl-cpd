from dataclasses import dataclass

from pysatl_cpd.labeled_data import LabeledCpdData


@dataclass
class OnlineCpdBenchmarkingResult:
    change_point: int
    delay: int


@dataclass
class DataInformation:
    data: LabeledCpdData
    configuration_name: str
    experiment_name: str
