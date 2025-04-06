import numpy as np
import pytest

from pysatl_cpd.core.algorithms.Entropies.entropies_alogorithms.shannon_entropy import ShannonEntropyAlgorithm


def set_seed():
    np.random.seed(1)


def construct_shannon_entropy_algorithm():
    return ShannonEntropyAlgorithm(
        window_size=40,
        step=20,
        bins=10,
        threshold=0.3
    )


@pytest.fixture(scope="function")
def data_params():
    return {
        "num_of_tests": 10,
        "size": 500,
        "change_point": 250,
        "tolerable_deviation": 30,
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
def outer_shannon_algorithm():
    return construct_shannon_entropy_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_shannon_entropy_algorithm()

    return _factory


def test_consecutive_detection(outer_shannon_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        result = outer_shannon_algorithm.detect(data)
        assert result > 0


def test_correctness_of_consecutive_detection(
    outer_shannon_algorithm, inner_algorithm_factory, generate_data, data_params
):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        inner_algorithm = inner_algorithm_factory()
        outer_result = outer_shannon_algorithm.detect(data)
        inner_result = inner_algorithm.detect(data)
        assert outer_result == inner_result


def test_consecutive_localization(outer_shannon_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        result = outer_shannon_algorithm.localize(data)
        assert len(result) > 0
        
        closest_point = min(result, key=lambda x: abs(x - data_params["change_point"]))
        assert (
            data_params["change_point"] - data_params["tolerable_deviation"]
            <= closest_point
            <= data_params["change_point"] + data_params["tolerable_deviation"]
        )


def test_correctness_of_consecutive_localization(
    outer_shannon_algorithm, inner_algorithm_factory, generate_data, data_params
):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        inner_algorithm = inner_algorithm_factory()
        outer_result = outer_shannon_algorithm.localize(data)
        inner_result = inner_algorithm.localize(data)
        assert outer_result == inner_result


def test_entropy_calculation():
    algorithm = construct_shannon_entropy_algorithm()
    
    uniform_probs = np.ones(8) / 8  
    computed_entropy = algorithm._compute_entropy(uniform_probs)
    expected_entropy = 3.0  
    
    assert abs(computed_entropy - expected_entropy) < 0.01
       
    
    certain_probs = np.zeros(8)
    certain_probs[0] = 1.0
    computed_entropy = algorithm._compute_entropy(certain_probs)
    expected_entropy = 0.0
    
    assert abs(computed_entropy - expected_entropy) < 0.01


def test_sliding_window():
    algorithm = ShannonEntropyAlgorithm(window_size=10, step=5, bins=5, threshold=0.3)
    
    data = np.concatenate([np.zeros(50), np.ones(50)])
    entropy_values = algorithm._sliding_entropy(data)
    
    expected_length = (len(data) - algorithm._window_size) // algorithm._step + 1
    assert len(entropy_values) == expected_length
    
    transition_window_index = 50 // algorithm._step - 1
    assert np.abs(entropy_values[transition_window_index+1] - entropy_values[transition_window_index]) > algorithm._threshold


def test_change_point_detection():
    algorithm = construct_shannon_entropy_algorithm()
    
    entropy_values = np.zeros(10)
    entropy_values[5] = 1.0  
    
    changes = algorithm._detect_change_points(entropy_values)
    assert len(changes) > 0
    assert 4 in changes or 5 in changes