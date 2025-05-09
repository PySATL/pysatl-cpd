import numpy as np
import pytest

from pysatl_cpd.core.algorithms.bayesian.detectors.threshold import ThresholdDetector
from pysatl_cpd.core.algorithms.bayesian.hazards.constant import ConstantHazard
from pysatl_cpd.core.algorithms.bayesian.likelihoods.gaussian_conjugate import (
    GaussianConjugate,
)
from pysatl_cpd.core.algorithms.bayesian.localizers.argmax import ArgmaxLocalizer
from pysatl_cpd.core.algorithms.bayesian_algorithm import BayesianAlgorithm


def set_seed():
    np.random.seed(1)


def construct_bayesian_algorithm():
    return BayesianAlgorithm(
        learning_steps=50,
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
                np.random.normal(
                    loc=5,
                    scale=2,
                    size=data_params["size"] - data_params["change_point"],
                ),
            ]
        )

    return _generate_data


@pytest.fixture(scope="function")
def outer_bayesian_algorithm():
    return construct_bayesian_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_bayesian_algorithm()

    return _factory


class TestBayesianAlgorithm:
    def test_consecutive_detection(self, outer_bayesian_algorithm, generate_data, data_params):
        for _ in range(data_params["num_of_tests"]):
            data = generate_data()
            result = outer_bayesian_algorithm.detect(data)
            assert result, "There was undetected change point in data"

    def test_correctness_of_consecutive_detection(
        self,
        outer_bayesian_algorithm,
        inner_algorithm_factory,
        generate_data,
        data_params,
    ):
        for _ in range(data_params["num_of_tests"]):
            data = generate_data()
            inner_algorithm = inner_algorithm_factory()
            outer_result = outer_bayesian_algorithm.detect(data)
            inner_result = inner_algorithm.detect(data)
            assert outer_result == inner_result, "Consecutive and independent detection should give same results"

    def test_consecutive_localization(self, outer_bayesian_algorithm, generate_data, data_params):
        for _ in range(data_params["num_of_tests"]):
            data = generate_data()
            result = outer_bayesian_algorithm.localize(data)
            assert (
                len(result) > 0
                and data_params["change_point"] - data_params["tolerable_deviation"]
                <= result[0]
                <= data_params["change_point"] + data_params["tolerable_deviation"]
            ), "Incorrect change point localization"

    def test_correctness_of_consecutive_localization(
        self,
        outer_bayesian_algorithm,
        inner_algorithm_factory,
        generate_data,
        data_params,
    ):
        for _ in range(data_params["num_of_tests"]):
            data = generate_data()
            inner_algorithm = inner_algorithm_factory()
            outer_result = outer_bayesian_algorithm.localize(data)
            inner_result = inner_algorithm.localize(data)
            assert outer_result == inner_result, "Consecutive and independent localization should give same results"
