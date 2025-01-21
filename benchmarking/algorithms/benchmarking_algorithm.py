from abc import ABC, abstractmethod

from CPDShell.Core.algorithms.abstract_algorithm import Algorithm
from benchmarking.benchmarking_info import AlgorithmBenchmarkingInfo


class BenchmarkingAlgorithm(ABC, Algorithm):
    @abstractmethod
    def get_benchmarking_info(self) -> AlgorithmBenchmarkingInfo:
        raise NotImplementedError
    
    @abstractmethod
    def get_metaparameters(self) -> dict[str, str]:
        raise NotImplementedError
