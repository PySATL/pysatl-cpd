import os
import typing as tp
from pathlib import Path
import logging
import datetime
import yaml

import numpy as np

from pysatl_cpd.core.algorithms.classification.test_statistics.threshold_overcome import ThresholdOvercome
from benchmarking.algorithms.benchmarking_knn import BenchmarkingKNNAlgorithm
from benchmarking.scrubber.benchmarking_linear_scrubber import BenchmarkingLinearScrubber
from benchmarking.report.benchmarking_report import BenchmarkingReport
from experiments.experiment import Experiment


def metric(obs1: np.ndarray | float, obs2: np.ndarray | float) -> float:
    return float(np.linalg.norm(obs1 - obs2))


K = 7
THRESHOLD = 3.0
INDENT_FACTOR = 0.25
WINDOW_SIZE = 48
SHIFT_FACTOR = 0.5
SIGNIFICANCE_LEVEL = 0.05
DELTA = 0.01
DATA_SIZE = 200
SAMPLE_COUNT = 1000
WITHOUT_CP_SAMPLE_COUNT = 1000
INTERVAL_LENGTH = int(WINDOW_SIZE / 4)
EXPECTED_CHANGE_POINTS = [100]
ROOT_DIR = Path()
ALG_NAME = "knn"
DISTR_CONFIG_PATH = ROOT_DIR / "experiments/distr_config_3.yaml"
DISTR_OPTIMIZATION_PATH = ROOT_DIR / "experiments/distr_optimization_1.yaml"
OPTIMAL_VALUES_PATH = ROOT_DIR / "benchmarking/optimal_values_5.yaml"

logger = logging.getLogger("BenchmarkInfo")

statistic_test = ThresholdOvercome(THRESHOLD)

cpd_algorithm = BenchmarkingKNNAlgorithm(metric, statistic_test, INDENT_FACTOR, K)
scrubber = BenchmarkingLinearScrubber(WINDOW_SIZE, SHIFT_FACTOR)

distribution_names_opt = ["0-exponential", "1-normal", "2-uniform", "3-beta", "4-weibull", "5-weibull", "6-weibull", "0-multivariate_normal", "1-multivariate_normal", "2-multivariate_normal"]
distribution_names = ["0-exponential-exponential", "1-normal-normal", "2-normal-normal", "3-normal-normal", "4-normal-normal", "5-normal-normal"]

dataset_dir = ROOT_DIR / f"experiments/datasets/"
without_cp_dir = dataset_dir / "without_cp"
results_dir = ROOT_DIR / f"experiments/experiment-{ALG_NAME}/results/"
dataset_dir.mkdir(parents=True, exist_ok=True)
without_cp_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)


Experiment.run_generator(DISTR_OPTIMIZATION_PATH, without_cp_dir, WITHOUT_CP_SAMPLE_COUNT)
Experiment.run_generator(DISTR_CONFIG_PATH, dataset_dir, SAMPLE_COUNT)
experiment = Experiment(cpd_algorithm, scrubber, logger)

for distr in distribution_names_opt:
    experiment.run_optimization(without_cp_dir / distr, results_dir / distr, OPTIMAL_VALUES_PATH, SIGNIFICANCE_LEVEL, DELTA, INTERVAL_LENGTH)

for distr in distribution_names:
    experiment.run_benchmark(dataset_dir / distr, OPTIMAL_VALUES_PATH, results_dir / distr, EXPECTED_CHANGE_POINTS, INTERVAL_LENGTH)

alg_metaparams = cpd_algorithm.get_metaparameters()
scrubber_metaparams = scrubber.get_metaparameters()
print(f"{alg_metaparams}")
print(f"{scrubber_metaparams}")

results = []

for distr in os.listdir(results_dir):
    if distr.count('-') != 2:
        continue

    with open(dataset_dir / distr / "config.yaml") as stream:
        # Gets distribution.
        distr_config = yaml.safe_load(stream)[0]["distributions"][0]

    with open(OPTIMAL_VALUES_PATH) as stream:
        optimal_values: list = yaml.safe_load(stream)

    threshold: float | None = None

    for v_conf in optimal_values:
        if (v_conf["config"]["algorithm"] == alg_metaparams
            and v_conf["config"]["scrubber"] == scrubber_metaparams
            and v_conf["distr_config"]["distributions"][0]["type"] == distr_config["type"]
            and v_conf["distr_config"]["distributions"][0]["parameters"] == distr_config["parameters"]):
            
            threshold = v_conf["optimal_values"]["threshold"]

    if threshold is not None:
        report = BenchmarkingReport(results_dir / distr, EXPECTED_CHANGE_POINTS, threshold, INTERVAL_LENGTH)
        report.add_power()
        results.append({"distr": distr, "power": report.get_result().filter_none()["power"], "threshold": str(threshold)})

result = sorted(results, key = lambda x: x["power"])
for distr_res in result:
    print(f"{distr_res["distr"]:<50} power: {distr_res["power"]:<23} threshold: {distr_res["threshold"]}")
