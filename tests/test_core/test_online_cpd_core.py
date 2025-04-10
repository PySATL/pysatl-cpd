import numpy as np
import pytest

from pysatl_cpd.core.algorithms.bayesian.detectors.threshold import ThresholdDetector
from pysatl_cpd.core.algorithms.bayesian.hazards.constant import ConstantHazard
from pysatl_cpd.core.algorithms.bayesian.likelihoods.gaussian_conjugate import GaussianConjugate
from pysatl_cpd.core.algorithms.bayesian.localizers.argmax import ArgmaxLocalizer
from pysatl_cpd.core.algorithms.bayesian_online_algorithm import BayesianOnline
from pysatl_cpd.core.online_cpd_core import OnlineCpdCore
from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider

DATA_PARAMS = {
    "num_of_tests": 10,
    "size": 500,
    "change_point": 250,
    "tolerable_deviation": 25,
}


@pytest.fixture(scope="session")
def data_params():
    return DATA_PARAMS


def construct_bayesian_online_algorithm():
    return BayesianOnline(
        learning_sample_size=50,
        likelihood=GaussianConjugate(),
        hazard=ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500))),
        detector=ThresholdDetector(threshold=0.04),
        localizer=ArgmaxLocalizer(),
    )


@pytest.fixture
def algorithm():
    return construct_bayesian_online_algorithm()


@pytest.fixture(params=[True, False], ids=["with_cp", "without_cp"])
def dataset(request, data_params):
    np.random.seed(42 + request.param_index)
    if request.param:
        return np.concatenate(
            [
                np.random.normal(0, 1, data_params["change_point"]),
                np.random.normal(5, 2, data_params["size"] - data_params["change_point"]),
            ]
        )
    return np.random.normal(0, 1, data_params["size"])


@pytest.fixture
def online_core(dataset):
    return OnlineCpdCore(
        algorithm=construct_bayesian_online_algorithm(), data_provider=ListUnivariateProvider(list(dataset))
    )


class TestOnlineCpdCore:
    @pytest.mark.parametrize("test_iteration", range(DATA_PARAMS["num_of_tests"]))
    @pytest.mark.parametrize("mode", ["detect", "localize"])
    def test_core_functionality(self, algorithm, online_core, dataset, data_params, mode, test_iteration):
        core_iterator = getattr(online_core, mode)()
        algo_method = getattr(algorithm, mode)

        for time_point in range(data_params["size"]):
            observation = dataset[time_point]
            algo_result = algo_method(observation)
            core_result = next(core_iterator)

            assert algo_result == core_result, (
                f"Different results at {time_point} between manual {mode} and core {mode} iteration"
            )
