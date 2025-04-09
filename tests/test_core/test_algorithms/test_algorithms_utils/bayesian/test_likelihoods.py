import numpy as np
import pytest

from pysatl_cpd.core.algorithms.bayesian.likelihoods.exponential_conjugate import ExponentialConjugate
from pysatl_cpd.core.algorithms.bayesian.likelihoods.gaussian_conjugate import GaussianConjugate


@pytest.fixture(scope="module")
def set_seed():
    np.random.seed(42)


@pytest.fixture(
    params=[
        (GaussianConjugate, {"pre_loc": 0, "pre_scale": 1, "post_loc": 5, "post_scale": 2}),
        (ExponentialConjugate, {"pre_scale": 1 / 0.5, "post_scale": 1 / 2}),
    ],
    ids=["Gaussian", "Exponential"],
)
def likelihood_config(request):
    return request.param


@pytest.fixture
def test_data(likelihood_config, set_seed):
    likelihood_cls, params = likelihood_config
    size = 500
    change_point = 250

    if likelihood_cls == GaussianConjugate:
        data = np.concatenate(
            [
                np.random.normal(params["pre_loc"], params["pre_scale"], change_point),
                np.random.normal(params["post_loc"], params["post_scale"], size - change_point),
            ]
        )
    else:
        data = np.concatenate(
            [
                np.random.exponential(params["pre_scale"], change_point),
                np.random.exponential(params["post_scale"], size - change_point),
            ]
        )
    return data


class TestConjugateLikelihood:
    @pytest.fixture(autouse=True)
    def setup(self, test_data, likelihood_config):
        self.likelihood_cls = likelihood_config[0]
        self.data = test_data
        self.size = 500
        self.change_point = 250
        self.learning_steps = 50

    def test_learning_and_update(self):
        likelihood = self.likelihood_cls()
        likelihood.learn(self.data[: self.learning_steps])

        metrics = {"after_learn": None, "before_cp": None, "after_cp": None}

        for time in range(self.learning_steps, self.size):
            observation = np.float64(self.data[time])
            pred_probs = likelihood.predict(observation)

            assert len(pred_probs) == time - self.learning_steps + 1

            current_mean = np.mean(pred_probs)
            if time == self.learning_steps + 1:
                metrics["after_learn"] = current_mean
            elif time == self.change_point - 1:
                metrics["before_cp"] = current_mean
            elif time == self.change_point + 1:
                metrics["after_cp"] = current_mean

            likelihood.update(observation)

        assert not np.isclose(metrics["after_learn"], metrics["before_cp"], atol=0.05)
        assert not np.isclose(metrics["before_cp"], metrics["after_cp"], atol=0.05)

    @pytest.mark.parametrize("data_size", [51, 100], ids=["small", "medium"])
    def test_clear(self, data_size):
        likelihood = self.likelihood_cls()
        test_data = self.data[:data_size]

        likelihood.learn(test_data[:-2])
        first = likelihood.predict(np.float64(test_data[-1]))

        likelihood.clear()
        likelihood.learn(test_data[:-2])
        second = likelihood.predict(np.float64(test_data[-1]))

        np.testing.assert_array_equal(first, second)
