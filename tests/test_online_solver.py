import numpy as np
import pytest

from pysatl_cpd.core.problem import CpdProblem
from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider
from pysatl_cpd.icpd_solver import CpdLocalizationResults
from pysatl_cpd.labeled_data import LabeledCpdData
from pysatl_cpd.online_cpd_solver import OnlineCpdSolver
from tests.test_core.test_algorithms.test_bayesian_online_algorithm import construct_bayesian_online_algorithm

DATA_PARAMS = {
    "num_tests": 10,
    "size": 500,
    "change_point": 250,
    "tolerable_deviation": 25,
}


@pytest.fixture(scope="session")
def data_params():
    return DATA_PARAMS


@pytest.fixture
def data_generator(data_params):
    def _generate(has_cp, test_iteration):
        seed = 42 + test_iteration
        np.random.seed(seed)
        if has_cp:
            return np.concatenate(
                [
                    np.random.normal(0, 1, data_params["change_point"]),
                    np.random.normal(5, 2, data_params["size"] - data_params["change_point"]),
                ]
            )
        return np.random.normal(0, 1, data_params["size"])

    return _generate


@pytest.fixture
def labeled_data_factory(data_params):
    def _factory(data, has_cp):
        return LabeledCpdData(raw_data=data, change_points=[data_params["change_point"]] if has_cp else None)

    return _factory


@pytest.fixture
def solver_factory():
    def _factory(data_input, with_localization):
        return OnlineCpdSolver(
            algorithm=construct_bayesian_online_algorithm(),
            algorithm_input=data_input,
            scenario=CpdProblem(with_localization),
        )

    return _factory


def pytest_generate_tests(metafunc):
    if "test_iteration" in metafunc.fixturenames:
        metafunc.parametrize("test_iteration", range(DATA_PARAMS["num_tests"]))


class TestOnlineCpdSolver:
    @pytest.mark.parametrize(
        "has_cp,with_localization,is_labeled",
        [
            (False, True, True),
            (True, True, True),
            (False, True, False),
            (True, True, False),
            (False, False, True),
            (True, False, True),
            (False, False, False),
            (True, False, False),
        ],
    )
    def test_all_scenarios(
        self,
        data_generator,
        labeled_data_factory,
        solver_factory,
        has_cp,
        with_localization,
        is_labeled,
        test_iteration,
        data_params,
    ):
        raw_data = data_generator(has_cp, test_iteration)

        data_input = labeled_data_factory(raw_data, has_cp) if is_labeled else ListUnivariateProvider(raw_data.tolist())

        solver = solver_factory(data_input, with_localization)
        result = solver.run()

        if with_localization:
            assert isinstance(result, CpdLocalizationResults), "Localization result must be CpdLocalizationResults"
            if has_cp:
                assert len(result.result) == 1, "There must be only one change point"
                assert abs(result.result[0] - data_params["change_point"]) <= data_params["tolerable_deviation"], (
                    "Change point must lie in tolerable interval"
                )
                if is_labeled:
                    assert result.expected_result == [data_params["change_point"]], (
                        "Labeled change point must be equal to generated one"
                    )
                else:
                    assert result.expected_result is None, "Expected result must be None for not labeled data"
            else:
                assert result.result == [], "There must be no change points"
        else:
            assert isinstance(result, int), "Detection result must be a number of detected change points"
            assert result == (1 if has_cp else 0), (
                "Number of change points must be equal to expected in the generated data"
            )
