from pathlib import Path
import logging
import datetime
import yaml
from shutil import rmtree
import pytest
import numpy as np

from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from benchmarking.worker.optimal_threshold import OptimalThresholdWorker
from benchmarking.worker.benchmarking import BenchmarkingWorker
from benchmarking.scrubber.benchmarking_linear_scrubber import BenchmarkingLinearScrubber
from benchmarking.algorithms.benchmarking_knn import BenchmarkingKNNAlgorithm
from benchmarking.algorithms.benchmarking_classification import BenchmarkingClassificationAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.decision_tree.decision_tree_classifier import DecisionTreeClassifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.quality_metrics.classification.mcc import MCC
from CPDShell.shell import CPDProblem
from CPDShell.labeled_data import LabeledCPData
from benchmarking.report.benchmarking_report import BenchmarkingReport, Measures
from benchmarking.generator.generator import DistributionGenerator, Distribution, DistributionType, DistributionComposition
from benchmarking.worker.performance import PerformanceWorker


def metric(obs1: np.ndarray | float, obs2: np.ndarray | float) -> float:
    return np.linalg.norm(obs1 - obs2)


THRESHOLD = 3.0
INDENT_FACTOR = 0.25
WINDOW_SIZE = 48
SHIFT_FACTOR = 0.5
K = 7
ROOT_DIR = Path()
DATASET_PATH = "/home/artemii/dev/PySATL-CPD-Module/experiments/datasets/0-50-10000/sample_0"


@pytest.mark.benchmark(
    min_rounds=5,
    min_time=5.0,
    warmup=True,
    warmup_iterations=1,
    disable_gc=True
)
def test_bench(benchmark):
    # print(DATASET_PATH.absolute())
    # alg = BenchmarkingKNNAlgorithm(metric, ThresholdOvercome(THRESHOLD), INDENT_FACTOR, K)
    alg = BenchmarkingClassificationAlgorithm(DecisionTreeClassifier(), MCC(), ThresholdOvercome(1.0), INDENT_FACTOR)
    scrubber = BenchmarkingLinearScrubber(WINDOW_SIZE, SHIFT_FACTOR)
    data = list(LabeledCPData.read_generated_datasets(DATASET_PATH).values())[0].raw_data
    shell = CPDProblem(data, cpd_algorithm=alg, scrubber=scrubber)
    benchmark(shell.run_cpd)
