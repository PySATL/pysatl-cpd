import numpy as np
import pytest

from pysatl_cpd.core.algorithms.bayesian.detectors.threshold import ThresholdDetector
from pysatl_cpd.core.algorithms.bayesian.hazards.constant import ConstantHazard
from pysatl_cpd.core.algorithms.bayesian.likelihoods.exponential_conjugate import ExponentialConjugate
from pysatl_cpd.core.algorithms.bayesian.likelihoods.gaussian_conjugate import (
    GaussianConjugate,
)
from pysatl_cpd.core.algorithms.bayesian.likelihoods.heuristic_gaussian_vs_exponential import (
    HeuristicGaussianVsExponential,
)
from pysatl_cpd.core.algorithms.bayesian.localizers.argmax import ArgmaxLocalizer
from pysatl_cpd.core.algorithms.bayesian_online_algorithm import BayesianOnline


@pytest.fixture(scope="module")
def set_seed():
    np.random.seed(1)


@pytest.fixture(params=["normal", "exponential"], ids=["normal", "exponential"])
def distribution_type(request):
    return request.param


@pytest.fixture
def algorithm_factory(distribution_type):
    def _factory():
        likelihood = None
        match distribution_type:
            case "normal":
                likelihood = GaussianConjugate()
            case "exponential":
                likelihood = ExponentialConjugate()
            case "heuristic":
                likelihood = HeuristicGaussianVsExponential()
            case _:
                raise ValueError("Unsupported likelihood")

        return BayesianOnline(
            learning_sample_size=5,
            likelihood=likelihood,
            hazard=ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500))),
            detector=ThresholdDetector(threshold=0.04),
            localizer=ArgmaxLocalizer(),
        )

    return _factory


@pytest.fixture
def generate_data(distribution_type, data_params):
    def _generate():
        np.random.seed(1)
        cp = data_params["change_point"]
        size = data_params["size"]

        match distribution_type:
            case "normal":
                return np.concatenate([np.random.normal(0, 1, cp), np.random.normal(5, 2, size - cp)])
            case "exponential":
                return np.concatenate([np.random.exponential(1.0, cp), np.random.exponential(0.5, size - cp)])
            case "heuristic":
                return np.concatenate([np.random.exponential(1.0, cp), np.random.normal(5, 2, size - cp)])
            case _:
                raise ValueError("Unsupported likelihood")

    return _generate


@pytest.fixture(scope="function")
def data_params(distribution_type):
    base_params = {
        "num_of_tests": 10,
        "size": 500,
        "change_point": 250,
        "tolerable_deviation": 25,
    }
    return base_params


@pytest.mark.parametrize("distribution_type", ["normal", "exponential", "heuristic"])
class TestBayesianOnlineAlgorithm:
    @pytest.fixture(autouse=True)
    def setup(self, algorithm_factory):
        self.algorithm_factory = algorithm_factory

    def test_consecutive_detection(self, distribution_type, generate_data, data_params):
        for _ in range(data_params["num_of_tests"]):
            outer_algorithm = self.algorithm_factory()
            data = generate_data()
            was_change_point = False
            for value in data:
                result = outer_algorithm.detect(value)
                if result:
                    was_change_point = True

            assert was_change_point, "There was undetected change point in data"

    def test_correctness_of_consecutive_detection(self, generate_data, data_params):
        outer_algorithm = self.algorithm_factory()
        for _ in range(data_params["num_of_tests"]):
            data = generate_data()
            inner_algorithm = self.algorithm_factory()
            outer_result = []
            inner_result = []

            for value in data:
                outer_result.append(outer_algorithm.detect(value))
                inner_result.append(inner_algorithm.detect(value))

            outer_algorithm.clear()
            assert outer_result == inner_result, "Consecutive and independent detection should give same results"

    def test_consecutive_localization(self, generate_data, data_params):
        outer_algorithm = self.algorithm_factory()
        for _ in range(data_params["num_of_tests"]):
            data = generate_data()
            was_change_point = False

            for value in data:
                result = outer_algorithm.localize(value)
                if result is None:
                    continue

                was_change_point = True

                assert (
                    data_params["change_point"] - data_params["tolerable_deviation"]
                    <= result
                    <= data_params["change_point"] + data_params["tolerable_deviation"]
                ), "Incorrect change point localization"

            outer_algorithm.clear()
            assert was_change_point, "Actual change point was not detected at all"

    def test_correctness_of_consecutive_localization(self, generate_data, data_params):
        outer_algorithm = self.algorithm_factory()
        for _ in range(data_params["num_of_tests"]):
            data = generate_data()
            inner_algorithm = self.algorithm_factory()

            for value in data:
                outer_result = outer_algorithm.localize(value)
                inner_result = inner_algorithm.localize(value)
                assert outer_result == inner_result, "Consecutive and independent localization should give same results"

            outer_algorithm.clear()

    def test_online_detection_correctness(self, generate_data, data_params):
        for _ in range(data_params["num_of_tests"]):
            algorithm = self.algorithm_factory()
            data = generate_data()
            for time, value in np.ndenumerate(data):
                result = algorithm.detect(value)
                if result:
                    assert data_params["change_point"] <= time[0], "Change point cannot be detected beforehand"

    def test_online_localization_correctness(self, generate_data, data_params):
        for _ in range(data_params["num_of_tests"]):
            algorithm = self.algorithm_factory()
            data = generate_data()
            for time, value in np.ndenumerate(data):
                result = algorithm.localize(value)
                if result:
                    assert result <= time[0], "Change point cannot be in future"
                    assert data_params["change_point"] <= time[0], "Change point cannot be detected beforehand"
