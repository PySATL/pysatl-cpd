import os
import typing as tp
from pathlib import Path
import logging
import datetime
import yaml

from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
# from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.svm.svm_classifier import SVMClassifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.rf.rf_classifier import RFClassifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.decision_tree.decision_tree_classifier import DecisionTreeClassifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.quality_metrics.classification.mcc import MCC
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from benchmarking.algorithms.benchmarking_classification import BenchmarkingClassificationAlgorithm
from benchmarking.algorithms.benchmarking_knn import BenchmarkingKNNAlgorithm
from benchmarking.scrubber.benchmarking_linear_scrubber import BenchmarkingLinearScrubber
from benchmarking.report.benchmarking_report import BenchmarkingReport
from benchmarking.report.benchmarking_report import Measures
from experiments.experiment import Experiment


def metric(obs1: float, obs2: float) -> float:
    return abs(obs1 - obs2)


K = 7
THRESHOLD = 3.0
INDENT_FACTOR = 0.25
WINDOW_SIZE = 48
SHIFT_FACTOR = 0.5
# SIGNIFICANCE_LEVEL = 0.03
DELTA = 0.005
DATA_SIZE = 200
SAMPLE_COUNT = 10000
WITHOUT_CP_SAMPLE_COUNT = 10000
INTERVAL_LENGTH = int(WINDOW_SIZE / 4)
EXPECTED_CHANGE_POINTS = [100]
ROOT_DIR = Path()
DISTR_CONFIG_PATH = ROOT_DIR / "experiments/distr_config_2.yaml"
DISTR_OPTIMIZATION_PATH = ROOT_DIR / "experiments/distr_optimization.yaml"
OPTIMAL_VALUES_PATH = ROOT_DIR / "benchmarking/optimal_values.yaml"
QUALITY_METRIC = MCC()
EXP_N = 7

logger = logging.getLogger("BenchmarkInfo")
fileHandler = logging.FileHandler(f"{ROOT_DIR}/experiments/benchmark_info.log", mode="a", encoding="utf-8")
logger.addHandler(fileHandler)
logger.setLevel("INFO")
logger.info(datetime.datetime.now())

statistic_test = ThresholdOvercome(THRESHOLD)

for alg_name in ["rf"]:
    # for k in [7]:
    significance_level = 0.05
    # if alg_name == "svm":
    #     cpd_algorithm = BenchmarkingClassificationAlgorithm(SVMClassifier("linear"), QUALITY_METRIC, statistic_test, INDENT_FACTOR)
    #     logger.info("SVM Classifier")
    #     logger.info(f"Metric: {type(QUALITY_METRIC).__name__}")
    # if alg_name == "knn":
    # cpd_algorithm = BenchmarkingKNNAlgorithm(metric, statistic_test, INDENT_FACTOR, K)
    # logger.info("KNN Algorithm")
    # logger.info(f"K: {K}")
    # if alg_name == "rf":
    cpd_algorithm = BenchmarkingClassificationAlgorithm(RFClassifier(), QUALITY_METRIC, statistic_test, INDENT_FACTOR)
    logger.info("RF Classifier")
    logger.info(f"Metric: {type(QUALITY_METRIC).__name__}")
    # if alg_name == "dt":
    #     cpd_algorithm = BenchmarkingClassificationAlgorithm(DecisionTreeClassifier(), QUALITY_METRIC, statistic_test, INDENT_FACTOR)
    #     logger.info("DT Classifier")
    #     logger.info(f"Metric: {type(QUALITY_METRIC).__name__}")
    # distribution_names = ["0-exponential-exponential", "1-normal-normal"]
    # distribution_names = ["2-normal-normal", "3-normal-normal"]
    # distribution_names = ["4-normal-normal", "5-normal-normal"]
    # distribution_names = ["6-uniform-uniform", "7-uniform-uniform"]
    # distribution_names = ["8-uniform-uniform", "9-uniform-uniform"]
    # distribution_names = ["10-uniform-uniform", "11-beta-beta"]
    # distribution_names = ["12-beta-beta", "13-weibull-weibull"]
    # distribution_names = ["14-weibull-weibull", "15-weibull-weibull"]
    # distribution_names = ["16-weibull-weibull", "17-weibull-weibull"]
    distribution_names = ["18-weibull-weibull"]
    # distribution_names = ["4-normal-normal", "5-normal-normal", "6-uniform-uniform", "7-uniform-uniform"]
    # distribution_names = ["8-uniform-uniform", "9-uniform-uniform", "10-uniform-uniform", "11-beta-beta"]
    # distribution_names = ["12-beta-beta", "13-weibull-weibull", "14-weibull-weibull", "15-weibull-weibull"]
    # distribution_names = ["16-weibull-weibull", "17-weibull-weibull", "18-weibull-weibull"]
    # distribution_names = ["0-exponential", "1-normal", "2-uniform", "3-beta", "4-weibull", "5-weibull", "6-weibull"]
    # distribution_names = "0-exponential"
    # distribution_names = "1-normal"
    # distribution_names = "2-uniform"
    # distribution_names = "3-beta"
    # distribution_names = "4-weibull"
    # distribution_names = "5-weibull"
    # distribution_names = "6-weibull"
    dataset_dir = ROOT_DIR / f"experiments/experiment-knn-3/dataset/"
    without_cp_dir = dataset_dir / "without_cp"
    results_dir = ROOT_DIR / f"experiments/experiment-{alg_name}-{EXP_N}/results/"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    without_cp_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    scrubber = BenchmarkingLinearScrubber(WINDOW_SIZE, SHIFT_FACTOR)
    # logger.info(f"Window length: {WINDOW_SIZE}")

    # Experiment.run_generator(DISTR_OPTIMIZATION_PATH, without_cp_dir, WITHOUT_CP_SAMPLE_COUNT)
    # # Experiment.generate_without_cp(DISTR_CONFIG_PATH, without_cp_dir, SAMPLE_COUNT)
    # Experiment.run_generator(DISTR_CONFIG_PATH, dataset_dir, SAMPLE_COUNT)

    experiment = Experiment(cpd_algorithm, scrubber, logger)
    # for without_cp_dataset in os.listdir(without_cp_dir):
    #     if not os.path.isdir(without_cp_dir / without_cp_dataset):
    #         continue

    # experiment.run_optimization(without_cp_dir / distribution_names, results_dir / distribution_names, OPTIMAL_VALUES_PATH, significance_level, DELTA, INTERVAL_LENGTH)

    # for distr in os.listdir(dataset_dir):
    #     if distr.count('-') != 2:
    #         continue

    #     experiment.run_benchmark(dataset_dir / distr, OPTIMAL_VALUES_PATH, results_dir / distr, EXPECTED_CHANGE_POINTS, INTERVAL_LENGTH)

        
    for distr in distribution_names:
        experiment.run_benchmark(dataset_dir / distr, OPTIMAL_VALUES_PATH, results_dir / distr, EXPECTED_CHANGE_POINTS, INTERVAL_LENGTH)

    # for distr in os.listdir(results_dir):
    #     if distr.count('-') != 2:
    #         continue

    #     with open(dataset_dir / distr / "config.yaml") as stream:
    #         # Gets distribution.
    #         distr_config = yaml.safe_load(stream)[0]["distributions"][0]

    #     with open(OPTIMAL_VALUES_PATH) as stream:
    #         optimal_values: list = yaml.safe_load(stream)

    #     threshold: float | None = None
    #     alg_metaparams = cpd_algorithm.get_metaparameters()
    #     scrubber_metaparams = scrubber.get_metaparameters()

    #     for v_conf in optimal_values:
    #         if (v_conf["config"]["algorithm"] == alg_metaparams
    #             and v_conf["config"]["scrubber"] == scrubber_metaparams
    #             and v_conf["distr_config"]["distributions"][0]["type"] == distr_config["type"]
    #             and v_conf["distr_config"]["distributions"][0]["parameters"] == distr_config["parameters"]):
                
    #             threshold = v_conf["optimal_values"]["threshold"]

    #     if threshold is not None:
    #         report = BenchmarkingReport(results_dir / distr, EXPECTED_CHANGE_POINTS, threshold, INTERVAL_LENGTH)
    #         report.add_power()
    #         # report.add_scrubbing_alg_info()
    #         print(f"{distr}: {report.get_result().filter_out_none()}")
        