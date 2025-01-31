import typing as tp
from pathlib import Path
import logging
import datetime

from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.svm.svm_classifier import SVMClassifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.rf.rf_classifier import RFClassifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.decision_tree.decision_tree_classifier import DecisionTreeClassifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.quality_metrics.classification.mcc import MCC
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from benchmarking.scrubber.benchmarking_linear_scrubber import BenchmarkingLinearScrubber
from benchmarking.algorithms.benchmarking_knn import BenchmarkingKNNAlgorithm
from benchmarking.worker.optimal_threshold_worker import OptimalThresholdWorker
from benchmarking.worker.benchmarking_worker import BenchmarkingKNNWorker
from benchmarking.scrubber.benchmarking_linear_scrubber import BenchmarkingLinearScrubber
from experiments.experiment import Experiment


def metric(obs1: float, obs2: float) -> float:
    return abs(obs1 - obs2)


K = 5
THRESHOLD = 3.0
INDENT_FACTOR = 0.25
WINDOW_SIZE = 40
SHIFT_FACTOR = 0.5
SIGNIFICANCE_LEVEL = 0.03
DELTA = 0.005
DATA_SIZE = 200
SAMPLE_COUNT = 100
INTERVAL_LENGTH = WINDOW_SIZE / 4
EXPECTED_CHANGE_POINTS = [100]
ROOT_DIR = Path()
DISTR_CONFIG_PATH = ROOT_DIR / "experiments/distr_config_2.yaml"
DISTR_OPTIMIZATION_PATH = ROOT_DIR / "experiments/distr_optimization.yaml"
OPTIMAL_VALUES_PATH = ROOT_DIR / "benchmarking/optimal_values.yaml"
QUALITY_METRIC = MCC()
EXP_N = 1

logger = logging.getLogger("BenchmarkInfo")
fileHandler = logging.FileHandler(f"{ROOT_DIR}/experiments/benchmark_info.log", mode="a", encoding="utf-8")
logger.addHandler(fileHandler)
logger.setLevel("INFO")
logger.info(datetime.datetime.now())

statistic_test = ThresholdOvercome(THRESHOLD)

for alg_name in ["knn"]:
    # for k in [7]:
    significance_level = 0.05
    # if alg_name == "svm":
    #     cpd_algorithm = ClassificationAlgorithm(SVMClassifier(), QUALITY_METRIC, statistic_test, INDENT_FACTOR)
    #     logger.info("SVM Classifier")
    #     logger.info(f"Metric: {type(QUALITY_METRIC).__name__}")
    if alg_name == "knn":
        cpd_algorithm = BenchmarkingKNNAlgorithm(metric, statistic_test, INDENT_FACTOR, K)
        logger.info("KNN Algorithm")
        logger.info(f"K: {K}")
    # elif alg_name == "rf":
    #     cpd_algorithm = ClassificationAlgorithm(RFClassifier(), QUALITY_METRIC, statistic_test, INDENT_FACTOR)
    #     logger.info("RF Classifier")
    #     logger.info(f"Metric: {type(QUALITY_METRIC).__name__}")
    # elif alg_name == "dt":
    #     cpd_algorithm = ClassificationAlgorithm(DecisionTreeClassifier(), QUALITY_METRIC, statistic_test, INDENT_FACTOR)
    #     logger.info("DT Classifier")
    #     logger.info(f"Metric: {type(QUALITY_METRIC).__name__}")

    dataset_dir = ROOT_DIR / f"experiments/experiment-{alg_name}-{EXP_N}/dataset/"
    without_cp_dir = dataset_dir / "without_cp/0-exponential"
    results_dir = ROOT_DIR / f"experiments/experiment-{alg_name}-{EXP_N}/results/"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    without_cp_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    scrubber = BenchmarkingLinearScrubber(WINDOW_SIZE, SHIFT_FACTOR)
    logger.info(f"Window length: {WINDOW_SIZE}")

    # Experiment.run_generator(DISTR_OPTIMIZATION_PATH, without_cp_dir, SAMPLE_COUNT)
    # Experiment.generate_without_cp(DISTR_CONFIG_PATH, without_cp_dir, SAMPLE_COUNT)
    # Experiment.run_generator(DISTR_CONFIG_PATH, dataset_dir, SAMPLE_COUNT)

    experiment = Experiment(cpd_algorithm, scrubber, logger)
    # experiment.run_optimization(without_cp_dir, results_dir / "0-exponential", OPTIMAL_VALUES_PATH, significance_level, DELTA, INTERVAL_LENGTH)
    experiment.run_benchmark(dataset_dir / "0-exponential-exponential", OPTIMAL_VALUES_PATH, results_dir / "0-exponential-exponential", EXPECTED_CHANGE_POINTS, INTERVAL_LENGTH)
