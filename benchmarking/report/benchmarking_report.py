from pathlib import Path
from dataclasses import dataclass
import yaml
import sys
from benchmarking.worker.common.utils import Utils


# User can filter out none values and use only some.
# Actually, Measures can be class with method to incalsulate this work.
class Measures():
    def __init__(self) -> None:
        self.average_overall_time: float | None = None
        self.average_window_time: float | None = None
        self.memory: float | None = None # TODO: There are different types of memory
        self.power: float | None = None
        self.f1: float | None = None
        self.sl: float | None = None
        self.interval: int | None = None
        self.scrubbing_alg_info: dict[str, dict[str, str]] | None = None

    def filter_out_none(self) -> dict[str, float]:
        return {k: v for k, v in vars(self).items() if v is not None}


class BenchmarkingReport():
    def __init__(self, resultsDir: Path) -> None:
        self.__resultsDir = resultsDir
        self.__result: Measures = Measures()
        self.__sample_dirs = Utils.get_all_stats_dirs(resultsDir)

    def count_average_overall_time(self) -> None:
        overall_time = 0

        for sample_dir in self.__sample_dirs:
            with open(sample_dir / "benchmarking_info.yaml") as infile:
                # TODO: Unsure about type.
                loaded_info: dict[str, float] = yaml.safe_load(infile)

            overall_time += loaded_info["overall_time"]
        
        self.__result.average_overall_time = overall_time / len(self.__sample_dirs)

    def count_average_window_time(self) -> None:
        overall_time = 0

        for sample_dir in self.__sample_dirs:
            with open(sample_dir / "benchmarking_info.yaml") as infile:
                # TODO: Unsure about type.
                loaded_info: dict[str, float] = yaml.safe_load(infile)

            overall_time += loaded_info["average_time"]
        
        self.__result.average_window_time = overall_time / len(self.__sample_dirs)

    def count_memory(self) -> None:
        raise NotImplementedError

    def count_power(self) -> None:
        raise NotImplementedError

    def count_F1(self) -> None:
        raise NotImplementedError

    def count_SL(self) -> None:
        raise NotImplementedError

    def count_interval(self) -> None:
        raise NotImplementedError

    def add_scrubbing_alg_info(self) -> None:
        with open(self.__resultsDir / "config.yaml") as infile:
            # TODO: Unsure about type.
            loaded_info: dict[str, dict[str, str]] = yaml.safe_load(infile)
        
        self.__result.scrubbing_alg_info = loaded_info

    # Either serialized format or just string. But probably we need to count some statistics over results, so it would be good to make it serialized to deserialized and get some metrics.
    # We do not need to serialize it. We can just return some dataclass containig. Products do not need to have general interface. What if we do not need some metrics?
    # Make fields optional? Yeah.
    def get_result(self) -> Measures:
        return self.__result
