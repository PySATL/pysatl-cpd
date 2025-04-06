import numpy as np
import pytest

from pysatl_cpd.core.algorithms.Entropies.entropies_alogorithms.conditional_entropy import ConditionalEntropyAlgorithm


def set_seed():
    np.random.seed(1)


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
        data_x = np.concatenate(
            [
                np.random.normal(loc=0, scale=1, size=data_params["change_point"]),
                np.random.normal(loc=5, scale=2, size=data_params["size"] - data_params["change_point"]),
            ]
        )
        
        data_y = np.concatenate(
            [
                np.random.normal(loc=2, scale=1, size=data_params["change_point"]),
                np.random.normal(loc=-2, scale=1, size=data_params["size"] - data_params["change_point"]),
            ]
        )
        
        return data_x, data_y

    return _generate_data


@pytest.fixture
def conditional_algorithm_factory():
    def _factory(conditional_data):
        return ConditionalEntropyAlgorithm(
            conditional_data=conditional_data,
            window_size=40,
            step=20,
            bins=10,
            threshold=0.3
        )

    return _factory


def test_detection(conditional_algorithm_factory, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data_x, data_y = generate_data()
        algorithm = conditional_algorithm_factory(data_y)
        result = algorithm.detect(data_x)
        assert result > 0


def test_localization(conditional_algorithm_factory, generate_data, data_params):
    for _ in range(data_params["num_of_tests"]):
        data_x, data_y = generate_data()
        algorithm = conditional_algorithm_factory(data_y)
        result = algorithm.localize(data_x)
        assert len(result) > 0
        
        closest_point = min(result, key=lambda x: abs(x - data_params["change_point"]))
        assert (
            data_params["change_point"] - data_params["tolerable_deviation"]
            <= closest_point
            <= data_params["change_point"] + data_params["tolerable_deviation"]
        )


def test_conditional_entropy_calculation():
    data_x = np.random.normal(0, 1, 100)
    
    data_y1 = data_x + np.random.normal(0, 0.1, 100)  
    
    data_y2 = np.random.normal(0, 1, 100)  
    
    algorithm1 = ConditionalEntropyAlgorithm(conditional_data=data_y1, bins=10)
    algorithm2 = ConditionalEntropyAlgorithm(conditional_data=data_y2, bins=10)
    
    entropy1 = algorithm1._compute_conditional_entropy(data_x, data_y1)
    entropy2 = algorithm2._compute_conditional_entropy(data_x, data_y2)
    
    assert entropy1 < entropy2


def test_value_errors():
    algorithm1 = ConditionalEntropyAlgorithm(conditional_data=None)
    with pytest.raises(ValueError):
        algorithm1.detect(np.random.normal(0, 1, 100))
    
    algorithm2 = ConditionalEntropyAlgorithm(conditional_data=np.random.normal(0, 1, 50))
    with pytest.raises(ValueError):
        algorithm2.detect(np.random.normal(0, 1, 100))


def test_sliding_window():
    size = 100
    change_point = 50
    
    x1 = np.random.normal(0, 1, change_point)
    y1 = x1 + np.random.normal(0, 0.1, change_point)
    
    x2 = np.random.normal(0, 1, size - change_point)
    y2 = np.random.normal(0, 1, size - change_point)
    
    data_x = np.concatenate([x1, x2])
    data_y = np.concatenate([y1, y2])
    
    algorithm = ConditionalEntropyAlgorithm(
        conditional_data=data_y,
        window_size=10,
        step=5,
        bins=5,
        threshold=0.3
    )
    
    entropy_values = algorithm._sliding_conditional_entropy(data_x, data_y)
    
    expected_length = (len(data_x) - algorithm._window_size) // algorithm._step + 1
    assert len(entropy_values) == expected_length
    
    change_index = change_point // algorithm._step
    differences = np.abs(np.diff(entropy_values))
    
    significant_change = any(diff > algorithm._threshold for diff in differences[change_index-2:change_index+2])
    assert significant_change