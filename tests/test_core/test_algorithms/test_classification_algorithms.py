from itertools import product

import numpy as np
import numpy.typing as npt
import pytest

import pysatl_cpd.generator.distributions as dstr
from pysatl_cpd.core.algorithms.classification.classifiers.decision_tree import DecisionTreeClassifier
from pysatl_cpd.core.algorithms.classification.classifiers.knn import KNNClassifier
from pysatl_cpd.core.algorithms.classification.classifiers.rf import RFClassifier
from pysatl_cpd.core.algorithms.classification.classifiers.svm import SVMClassifier
from pysatl_cpd.core.algorithms.classification.quality_metrics.classification.f1 import F1
from pysatl_cpd.core.algorithms.classification.quality_metrics.classification.mcc import MCC
from pysatl_cpd.core.algorithms.classification.test_statistics.threshold_overcome import ThresholdOvercome
from pysatl_cpd.core.algorithms.classification_algorithm import ClassificationAlgorithm
from pysatl_cpd.core.algorithms.knn_algorithm import KNNAlgorithm
from pysatl_cpd.core.scrubber.data_providers import LabeledDataProvider
from pysatl_cpd.core.scrubber.linear import LinearScrubber
from pysatl_cpd.cpd_solver import CpdProblem, CpdSolver
from pysatl_cpd.labeled_data import LabeledCpdData

K = 7
CM_THRESHOLD = 4.5
INDENT_COEFF = 0.25
SHIFT_FACTOR = 0.5
WINDOW_SIZE = 48
SIZE = 200
CP_N = 100
TOLERABLE_DEVIATION = WINDOW_SIZE / 2
EXPECTED_CP = 100
CLASSIFIERS = ["knn", "svm", "rf", "dt"]
METRICS = ["mcc"]


def assert_result(actual):
    def in_interval(cp):
        return EXPECTED_CP - TOLERABLE_DEVIATION <= cp <= EXPECTED_CP + TOLERABLE_DEVIATION

    assert (len(actual) > 0 and all(in_interval(cp) for cp in actual)), "Incorrect change point localization"


def build_classification_alg(classifier_name, metric_name):
    match metric_name:
        case "f1":
            quality_metric = F1()
            threshold = 0.85
        case "mcc":
            quality_metric = MCC()
            threshold = 0.85
        case _:
            raise NotImplementedError("No such metric yet.")

    match classifier_name:
        case "knn":
            classifier = KNNClassifier(K)
        case "svm":
            classifier = SVMClassifier()
        case "dt":
            classifier = DecisionTreeClassifier()
        case "rf":
            classifier = RFClassifier()
        case _:
            raise NotImplementedError("No such classifier yet.")

    return ClassificationAlgorithm(classifier=classifier,
                                quality_metric=quality_metric,
                                test_statistic=ThresholdOvercome(threshold),
                                indent_coeff=INDENT_COEFF)


def build_solver(alg, data):
    data_provider = LabeledDataProvider(LabeledCpdData(data, [EXPECTED_CP]))
    scrubber = LinearScrubber(data_provider, WINDOW_SIZE, SHIFT_FACTOR)
    return CpdSolver(CpdProblem(to_localize=True), algorithm=alg, algorithm_input=scrubber)


@pytest.fixture(scope="session")
def univariate_data():
    np.random.seed(1)
    left_distr = dstr.Distribution.from_str(
        str(dstr.Distributions.UNIFORM),
            {"min": "2.0", "max": "2.1"})
    right_distr = dstr.Distribution.from_str(
        str(dstr.Distributions.UNIFORM),
            {"min": "0.0", "max": "0.1"})
    return np.concatenate(
        [
            left_distr.scipy_sample(EXPECTED_CP),
            right_distr.scipy_sample(SIZE - EXPECTED_CP),
        ]
    )


@pytest.fixture(scope="session")
def multivariate_data():
    np.random.seed(1)
    left_distr = dstr.Distribution.from_str(
        str(dstr.Distributions.MULTIVARIATIVE_NORMAL),
            {"mean": str([0.0] * 10)})
    right_distr = dstr.Distribution.from_str(
        str(dstr.Distributions.MULTIVARIATIVE_NORMAL),
            {"mean": str([5.0] * 10)})
    return np.concatenate(
        [
            left_distr.scipy_sample(EXPECTED_CP),
            right_distr.scipy_sample(SIZE - EXPECTED_CP)
        ]
    )


class TestClassificationCpd:
    @pytest.mark.parametrize(
            "classifier_name, metric",
            list(product(CLASSIFIERS, METRICS)),
    )
    def test_classification_cpd_univariate(self, classifier_name, metric, univariate_data):
        alg = build_classification_alg(classifier_name, metric)
        solver = build_solver(alg, univariate_data)
        actual = solver.run().result
        assert_result(actual)

    @pytest.mark.parametrize(
            "classifier_name, metric",
            list(product(CLASSIFIERS, METRICS)),
    )
    def test_classification_cpd_multivariate(self, classifier_name, metric, multivariate_data):
        alg = build_classification_alg(classifier_name, metric)
        solver = build_solver(alg, multivariate_data)
        actual = solver.run().result
        assert_result(actual)


class TestKnnCpd:
    @pytest.fixture(scope="function")
    def knn_cpd_univariate(self):
        def metric(obs1: float, obs2: float) -> float:
            return abs(obs1 - obs2)

        return KNNAlgorithm(distance_func=metric,
                            test_statistic=ThresholdOvercome(CM_THRESHOLD),
                            indent_coeff=INDENT_COEFF,
                            k=K)

    @pytest.fixture(scope="function")
    def knn_cpd_multivariate(self):
        def metric(obs1: npt.NDArray[np.float64], obs2: npt.NDArray[np.float64]) -> float:
            return float(np.linalg.norm(obs1 - obs2))

        return KNNAlgorithm(distance_func=metric,
                        test_statistic=ThresholdOvercome(CM_THRESHOLD),
                        indent_coeff=INDENT_COEFF,
                        k=K)

    def test_knn_cpd_univariate(self, knn_cpd_univariate, univariate_data):
        solver = build_solver(knn_cpd_univariate, univariate_data)
        actual = solver.run().result
        assert_result(actual)

    def test_knn_cpd_multivariate(self, knn_cpd_multivariate, multivariate_data):
        solver = build_solver(knn_cpd_multivariate, multivariate_data)
        actual = solver.run().result
        assert_result(actual)
