import numpy as np
import pytest

from pysatl_cpd.core.algorithms.bayesian.detectors.threshold import ThresholdDetector
from pysatl_cpd.core.algorithms.bayesian.hazards.constant import ConstantHazard
from pysatl_cpd.core.algorithms.bayesian.likelihoods.gaussian_conjugate import (
    GaussianConjugate,
)
from pysatl_cpd.core.algorithms.bayesian.localizers.argmax import ArgmaxLocalizer
from pysatl_cpd.core.algorithms.bayesian_online_algorithm import BayesianOnline


def set_seed():
    np.random.seed(1)


def construct_bayesian_online_algorithm():
    return BayesianOnline(
        learning_sample_size=5,
        likelihood=GaussianConjugate(),
        hazard=ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500))),
        detector=ThresholdDetector(threshold=0.04),
        localizer=ArgmaxLocalizer(),
    )


@pytest.fixture(scope="function")
def data_params():
    return {
        "num_of_tests": 10,
        "size": 500,
        "change_point": 250,
        "tolerable_deviation": 25,
    }


@pytest.fixture
def generate_data(data_params):
    def _generate_data():
        set_seed()
        return np.concatenate(
            [
                np.random.normal(loc=0, scale=1, size=data_params["change_point"]),
                np.random.normal(loc=5, scale=2, size=data_params["size"] - data_params["change_point"]),
            ]
        )

    return _generate_data


@pytest.fixture(scope="function")
def outer_bayesian_algorithm():
    return construct_bayesian_online_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_bayesian_online_algorithm()

    return _factory


def test_consecutive_detection(outer_bayesian_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        was_change_point = False
        for value in data:
            result = outer_bayesian_algorithm.detect(value)
            if result:
                was_change_point = True

        assert was_change_point, "There was undetected change point in data"


def test_correctness_of_consecutive_detection(
    outer_bayesian_algorithm, inner_algorithm_factory, generate_data, data_params
):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        inner_algorithm = inner_algorithm_factory()
        outer_result = []
        inner_result = []

        for value in data:
            outer_result.append(outer_bayesian_algorithm.detect(value))
            inner_result.append(inner_algorithm.detect(value))

        outer_bayesian_algorithm.clear()
        assert outer_result == inner_result, "Consecutive and independent detection should give same results"


def test_consecutive_localization(outer_bayesian_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        was_change_point = False

        for value in data:
            result = outer_bayesian_algorithm.localize(value)
            if result is None:
                continue

            was_change_point = True

            assert (
                data_params["change_point"] - data_params["tolerable_deviation"]
                <= result
                <= data_params["change_point"] + data_params["tolerable_deviation"]
            ), "Incorrect change point localization"

        outer_bayesian_algorithm.clear()
        assert was_change_point, "Actual change point was not detected at all"


def test_correctness_of_consecutive_localization(
    outer_bayesian_algorithm, inner_algorithm_factory, generate_data, data_params
):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        inner_algorithm = inner_algorithm_factory()

        for value in data:
            outer_result = outer_bayesian_algorithm.localize(value)
            inner_result = inner_algorithm.localize(value)
            assert outer_result == inner_result, "Consecutive and independent localization should give same results"

        outer_bayesian_algorithm.clear()


def test_online_detection_correctness(inner_algorithm_factory, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        algorithm = inner_algorithm_factory()
        data = generate_data()
        for time, value in np.ndenumerate(data):
            result = algorithm.detect(value)
            if result:
                assert data_params["change_point"] <= time[0], "Change point cannot be detected beforehand"


def test_online_localization_correctness(inner_algorithm_factory, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        algorithm = inner_algorithm_factory()
        data = generate_data()
        for time, value in np.ndenumerate(data):
            result = algorithm.localize(value)
            if result:
                assert result <= time[0], "Change point cannot be in future"
                assert data_params["change_point"] <= time[0], "Change point cannot be detected beforehand"
