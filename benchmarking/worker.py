import time as t

from benchmarking.benchmarking_dataclasses import OnlineCpdBenchmarkingResult
from pysatl_cpd.core.online_cpd_core import OnlineCpdCore


class Worker:
    def __init__(self, online_cpd: OnlineCpdCore):
        self.__cpd = online_cpd

    def run(self):
        start = t.perf_counter()
        change_points = [cp for cp in self.__cpd.localize()]
        finish = t.perf_counter()
        cpd_time = finish - start

        benchmarking_results = [
            OnlineCpdBenchmarkingResult(change_point=cp, delay=time - cp)
            for time, cp in enumerate(change_points)
            if cp is not None
        ]

        return benchmarking_results, cpd_time
