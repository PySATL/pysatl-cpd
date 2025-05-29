import numpy as np
import pytest

from pysatl_cpd.core.algorithms.bayesian.detectors.threshold import ThresholdDetector
from pysatl_cpd.core.algorithms.bayesian.hazards.constant import ConstantHazard
from pysatl_cpd.core.algorithms.bayesian.likelihoods.heuristic_gaussian_vs_exponential import (
    HeuristicGaussianVsExponential,
)
from pysatl_cpd.core.algorithms.bayesian.localizers.argmax import ArgmaxLocalizer
from pysatl_cpd.core.algorithms.bayesian_linear_heuristic import BayesianLinearHeuristic
from pysatl_cpd.core.algorithms.bayesian_online_algorithm import BayesianOnline
from pysatl_cpd.core.problem import CpdProblem
from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider
from pysatl_cpd.online_cpd_solver import OnlineCpdSolver


def generate_no_change_exponential(rate, n=40000, seed=None):
    np.random.seed(seed)
    return np.random.exponential(scale=1 / rate, size=n)


def generate_no_change_normal(mean, std, n=40000, seed=None):
    np.random.seed(seed)
    return np.random.normal(loc=mean, scale=std, size=n)


def generate_change_exp_to_exp(rate1, rate2, change_point, n=40000, seed=None):
    np.random.seed(seed)
    part1 = np.random.exponential(scale=1 / rate1, size=change_point)
    part2 = np.random.exponential(scale=1 / rate2, size=n - change_point)
    return np.concatenate([part1, part2])


def generate_change_norm_to_norm(mean1, std1, mean2, std2, change_point, n=40000, seed=None):
    np.random.seed(seed)
    part1 = np.random.normal(loc=mean1, scale=std1, size=change_point)
    part2 = np.random.normal(loc=mean2, scale=std2, size=n - change_point)
    return np.concatenate([part1, part2])


def generate_change_exp_to_norm(rate, mean, std, change_point, n=40000, seed=None):
    np.random.seed(seed)
    part1 = np.random.exponential(scale=1 / rate, size=change_point)
    part2 = np.random.normal(loc=mean, scale=std, size=n - change_point)
    return np.concatenate([part1, part2])


def generate_change_norm_to_exp(mean, std, rate, change_point, n=40000, seed=None):
    np.random.seed(seed)
    part1 = np.random.normal(loc=mean, scale=std, size=change_point)
    part2 = np.random.exponential(scale=1 / rate, size=n - change_point)
    return np.concatenate([part1, part2])


@pytest.fixture
def setup_algorithm():
    base_algorithm = BayesianOnline(
        learning_sample_size=20,
        likelihood=HeuristicGaussianVsExponential(),
        hazard=ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500))),
        detector=ThresholdDetector(threshold=0.04),
        localizer=ArgmaxLocalizer(),
    )
    heuristic_algorithm = BayesianLinearHeuristic(
        algorithm=base_algorithm, time_before_duplicate_start=275, duplicate_preparation_time=225
    )
    return base_algorithm, heuristic_algorithm


@pytest.mark.parametrize(
    "data_generator, params, true_cp",
    [
        (generate_no_change_exponential, {"rate": 2.0}, None),
        (generate_no_change_normal, {"mean": 0.0, "std": 1.0}, None),
        (generate_change_exp_to_exp, {"rate1": 2.0, "rate2": 0.5, "change_point": 10000}, 10000),
        (
            generate_change_norm_to_norm,
            {"mean1": 0.0, "std1": 1.0, "mean2": 5.0, "std2": 1.0, "change_point": 15000},
            15000,
        ),
        (generate_change_exp_to_norm, {"rate": 2.0, "mean": 5.0, "std": 1.0, "change_point": 20000}, 20000),
        (generate_change_norm_to_exp, {"mean": 0.0, "std": 1.0, "rate": 0.5, "change_point": 25000}, 25000),
    ],
)
def test_cpd_detection(setup_algorithm, data_generator, params, true_cp):
    _, heuristic_algorithm = setup_algorithm

    data = data_generator(**params, n=40000, seed=42)
    data_provider = ListUnivariateProvider(list(data))

    solver_heuristic = OnlineCpdSolver(
        scenario=CpdProblem(True), algorithm=heuristic_algorithm, algorithm_input=data_provider
    )
    result_heuristic = solver_heuristic.run()

    if true_cp is None:
        print(result_heuristic.result)
        assert len(result_heuristic.result) < len(data) / 500, "There shouldn't be too much change points"
    else:
        assert any(true_cp - 25 <= cp <= true_cp + 25 for cp in result_heuristic.result), (
            f"No detected change point near {true_cp} in heuristic result"
        )


def test_time_comparison(setup_algorithm):
    base_algorithm, heuristic_algorithm = setup_algorithm

    data = generate_change_exp_to_exp(rate1=2.0, rate2=0.5, change_point=10000, n=40000, seed=42)
    data_provider = ListUnivariateProvider(list(data))

    solver_heuristic = OnlineCpdSolver(
        scenario=CpdProblem(True), algorithm=heuristic_algorithm, algorithm_input=data_provider
    )
    time_heuristic = solver_heuristic.run().time_sec

    solver_base = OnlineCpdSolver(scenario=CpdProblem(True), algorithm=base_algorithm, algorithm_input=data_provider)
    time_base = solver_base.run().time_sec

    print(time_heuristic, time_base)
    assert time_heuristic < time_base, f"Heuristic time ({time_heuristic}) >= base time ({time_base})"
