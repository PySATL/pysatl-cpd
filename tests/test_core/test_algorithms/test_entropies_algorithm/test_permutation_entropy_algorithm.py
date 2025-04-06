import numpy as np
import pytest

from pysatl_cpd.core.algorithms.Entropies.entropies_alogorithms.permutation_entropy import PermutationEntropyAlgorithm


def set_seed():
    np.random.seed(1)


def construct_permutation_entropy_algorithm():
    return PermutationEntropyAlgorithm(
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
        t1 = np.linspace(0, 10*np.pi, data_params["change_point"])
        data1 = np.sin(t1) + np.random.normal(0, 0.1, size=len(t1))
        
        data2 = np.random.normal(0, 1, size=data_params["size"] - data_params["change_point"])
        
        return np.concatenate([data1, data2])

    return _generate_data


@pytest.fixture(scope="function")
def outer_permutation_algorithm():
    return construct_permutation_entropy_algorithm()


@pytest.fixture
def inner_algorithm_factory():
    def _factory():
        return construct_permutation_entropy_algorithm()

    return _factory


def test_consecutive_detection(outer_permutation_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        result = outer_permutation_algorithm.detect(data)
        assert result > 0


def test_correctness_of_consecutive_detection(
    outer_permutation_algorithm, inner_algorithm_factory, generate_data, data_params
):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        inner_algorithm = inner_algorithm_factory()
        outer_result = outer_permutation_algorithm.detect(data)
        inner_result = inner_algorithm.detect(data)
        assert outer_result == inner_result


def test_consecutive_localization(outer_permutation_algorithm, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        result = outer_permutation_algorithm.localize(data)
        assert len(result) > 0
        
        closest_point = min(result, key=lambda x: abs(x - data_params["change_point"]))
        assert (
            data_params["change_point"] - data_params["tolerable_deviation"]
            <= closest_point
            <= data_params["change_point"] + data_params["tolerable_deviation"]
        )


def test_correctness_of_consecutive_localization(
    outer_permutation_algorithm, inner_algorithm_factory, generate_data, data_params
):
    for _ in range(data_params["num_of_tests"]):
        data = generate_data()
        inner_algorithm = inner_algorithm_factory()
        outer_result = outer_permutation_algorithm.localize(data)
        inner_result = inner_algorithm.localize(data)
        assert outer_result == inner_result


def test_permutation_entropy_calculation():
    algorithm = construct_permutation_entropy_algorithm()
    
    t = np.linspace(0, 4*np.pi, 100)
    deterministic_signal = np.sin(t)
    deterministic_entropy = algorithm._calculate_permutation_entropy(deterministic_signal)
    
    random_signal = np.random.normal(0, 1, 100)
    random_entropy = algorithm._calculate_permutation_entropy(random_signal)
    
    assert deterministic_entropy < random_entropy
    
    constant_signal = np.ones(100)
    constant_entropy = algorithm._calculate_permutation_entropy(constant_signal)
    assert constant_entropy == 0


def test_embedding_dimension_effect():
    t = np.linspace(0, 10*np.pi, 200)
    signal = np.sin(t) + 0.1 * np.random.normal(0, 1, 200)
    
    entropies = []
    for dim in range(2, 6):
        algorithm = PermutationEntropyAlgorithm(
            window_size=40,
            step=20,
            embedding_dimension=dim,
            time_delay=1,
            threshold=0.2
        )
        entropy = algorithm._calculate_permutation_entropy(signal)
        entropies.append(entropy)
    
    for i in range(len(entropies) - 1):
        assert entropies[i] <= entropies[i+1]
           


def test_time_delay_effect():
    t = np.linspace(0, 20*np.pi, 400)
    signal = np.sin(t)
    
    period_samples = int(400 / 20)  
    
    algorithm1 = PermutationEntropyAlgorithm(
        window_size=40,
        step=20,
        embedding_dimension=3,
        time_delay=1,  
        threshold=0.2
    )
    
    algorithm2 = PermutationEntropyAlgorithm(
        window_size=40,
        step=20,
        embedding_dimension=3,
        time_delay=period_samples,  
        threshold=0.2
    )
    
    entropy1 = algorithm1._calculate_permutation_entropy(signal)
    entropy2 = algorithm2._calculate_permutation_entropy(signal)
    
    assert abs(entropy2 - entropy1) > 0.1