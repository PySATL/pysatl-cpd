import numpy as np
import pytest

from pysatl_cpd.core.algorithms.Entropies.entropies_alogorithms.bubble_entropy import BubbleEntropyAlgorithm


def set_seed():
    np.random.seed(1)


def construct_bubble_entropy_algorithm():
    return BubbleEntropyAlgorithm(
        window_size=40,
        step=20,
        embedding_dimension=3,
        time_delay=1,
        threshold=0.2
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
        data1 = np.zeros(data_params["change_point"])
        for i in range(5, data_params["change_point"]):
            data1[i] = 0.6 * data1[i-1] - 0.3 * data1[i-2] + 0.1 * data1[i-3] + np.random.normal(0, 0.1)
        
        data2 = np.zeros(data_params["size"] - data_params["change_point"])
        for i in range(5, len(data2)):
            data2[i] = 0.2 * data2[i-1] + 0.7 * data2[i-2] + np.random.normal(0, 0.5)
        
        return np.concatenate([data1, data2])

    return _generate_data


@pytest.fixture(scope="function")
def outer_bubble_algorithm():
    return construct_bubble_entropy_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_bubble_entropy_algorithm()

    return _factory


def test_consecutive_detection(outer_bubble_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        result = outer_bubble_algorithm.detect(data)
        assert result > 0


def test_correctness_of_consecutive_detection(
    outer_bubble_algorithm, inner_algorithm_factory, generate_data, data_params
):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        inner_algorithm = inner_algorithm_factory()
        outer_result = outer_bubble_algorithm.detect(data)
        inner_result = inner_algorithm.detect(data)
        assert outer_result == inner_result


def test_consecutive_localization(outer_bubble_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        result = outer_bubble_algorithm.localize(data)
        assert len(result) > 0
        
        closest_point = min(result, key=lambda x: abs(x - data_params["change_point"]))
        assert (
            data_params["change_point"] - data_params["tolerable_deviation"]
            <= closest_point
            <= data_params["change_point"] + data_params["tolerable_deviation"]
        )

def test_correctness_of_consecutive_localization(
    outer_bubble_algorithm, inner_algorithm_factory, generate_data, data_params
):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        inner_algorithm = inner_algorithm_factory()
        outer_result = outer_bubble_algorithm.localize(data)
        inner_result = inner_algorithm.localize(data)
        assert outer_result == inner_result


def test_bubble_entropy_calculation():
    algorithm = construct_bubble_entropy_algorithm()
    
    t = np.linspace(0, 4*np.pi, 100)
    deterministic_signal = np.sin(t)
    deterministic_entropy = algorithm._calculate_bubble_entropy(deterministic_signal)
    
    random_signal = np.random.normal(0, 1, 100)
    random_entropy = algorithm._calculate_bubble_entropy(random_signal)
    
    assert abs(deterministic_entropy - random_entropy) > 0.1
    
    pe_m = algorithm._calculate_permutation_entropy(random_signal, algorithm._embedding_dimension)
    pe_m_plus_1 = algorithm._calculate_permutation_entropy(random_signal, algorithm._embedding_dimension + 1)
    
    expected_bubble = (pe_m_plus_1 - pe_m) / np.log(
        (algorithm._embedding_dimension + 1) / algorithm._embedding_dimension
    )
    
    calc_bubble = algorithm._calculate_bubble_entropy(random_signal)
    
    assert abs(expected_bubble - calc_bubble) < 0.01


def test_sliding_window():
    algorithm = BubbleEntropyAlgorithm(
        window_size=20,
        step=10,
        embedding_dimension=3,
        time_delay=1,
        threshold=0.2
    )
    
    size = 100
    change_point = 50
    
    t1 = np.linspace(0, 5*np.pi, change_point)
    data1 = np.sin(t1)
    
    t2 = np.linspace(0, 5*np.pi, size - change_point)
    data2 = np.sin(t2) + np.random.normal(0, 0.5, size=len(t2))
    
    data = np.concatenate([data1, data2])
    
    entropy_values = algorithm._sliding_bubble_entropy(data)
    
    expected_length = (len(data) - algorithm._window_size) // algorithm._step + 1
    assert len(entropy_values) == expected_length
    
    change_index = (change_point - algorithm._window_size // 2) // algorithm._step
    
    changes = np.where(np.abs(np.diff(entropy_values)) > algorithm._threshold)[0]
    assert any(abs(change_index - idx) <= 2 for idx in changes)


def test_edge_cases():
    algorithm = construct_bubble_entropy_algorithm()
    
    short_signal = np.random.normal(0, 1, 10)
    entropy_short = algorithm._calculate_bubble_entropy(short_signal)
    assert not np.isnan(entropy_short)
    
    constant_signal = np.ones(50)
    entropy_constant = algorithm._calculate_bubble_entropy(constant_signal)
    assert entropy_constant == 0