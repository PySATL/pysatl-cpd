import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given

from pysatl_cpd.generator.changepoint_process import PoissonChangepointProcess
from pysatl_cpd.generator.distributions import Distribution


class MockDistribution(Distribution):
    def __init__(self, mean) -> None:
        self._mean = mean

    def scipy_sample(self, length: int) -> npt.NDArray[np.float64]:
        return np.array([self._mean] * length)

    @property
    def name(self) -> str:
        return super().name

    @property
    def params(self) -> dict[str, str]:
        return super().params

    @classmethod
    def from_params(cls, params: dict[str, str]) -> "Distribution":
        raise NotImplementedError


def mock_distribution_factory(mean):
    return MockDistribution(mean)


class TestPoissonChangepointProcess:
    def test_initialization_success(self):
        total_length = 1000
        mean_sampler = MockDistribution(mean=10.0)
        process = PoissonChangepointProcess(
            total_length=total_length,
            cp_intensity_per_point=0.01,
            mean_sampler=mean_sampler,
            distribution_factory=mock_distribution_factory,
        )
        assert process._total_length == total_length
        assert process._avg_segment_length == pytest.approx(100.0)
        assert process._mean_sampler is mean_sampler
        assert process._distribution_factory is mock_distribution_factory

    @pytest.mark.parametrize(
        "length, intensity",
        [
            (0, 0.1),
            (-100, 0.1),
            (100, 0),
            (100, -0.1),
            (0, 0),
        ],
    )
    def test_initialization_raises_value_error(self, length: int, intensity: float):
        with pytest.raises(ValueError):
            PoissonChangepointProcess(
                total_length=length,
                cp_intensity_per_point=intensity,
                mean_sampler=MockDistribution(mean=0),
                distribution_factory=mock_distribution_factory,
            )

    def test_generate_segments_output_structure(self):
        process = PoissonChangepointProcess(
            total_length=500,
            cp_intensity_per_point=0.1,
            mean_sampler=MockDistribution(mean=5.0),
            distribution_factory=mock_distribution_factory,
        )
        distributions, lengths = process.generate_segments()

        assert isinstance(distributions, list)
        assert isinstance(lengths, list)
        assert len(distributions) == len(lengths)
        assert all(isinstance(d, MockDistribution) for d in distributions)
        assert all(isinstance(length, int) for length in lengths)

    def test_distribution_creation_logic(self):
        fixed_mean = 42.0
        mean_sampler = MockDistribution(mean=fixed_mean)
        process = PoissonChangepointProcess(
            total_length=100,
            cp_intensity_per_point=0.2,
            mean_sampler=mean_sampler,
            distribution_factory=mock_distribution_factory,
        )
        distributions, _ = process.generate_segments()

        assert len(distributions) > 0
        for dist in distributions:
            assert isinstance(dist, MockDistribution)
            assert dist._mean == fixed_mean

    @given(
        total_length=st.integers(min_value=1, max_value=5000),
        cp_intensity=st.floats(min_value=0.001, max_value=1.0),
        mean_value=st.floats(min_value=-1e5, max_value=1e5),
    )
    def test_generate_segments_properties_with_hypothesis(
        self, total_length: int, cp_intensity: float, mean_value: float
    ):
        mean_sampler = MockDistribution(mean=mean_value)
        process = PoissonChangepointProcess(
            total_length=total_length,
            cp_intensity_per_point=cp_intensity,
            mean_sampler=mean_sampler,
            distribution_factory=mock_distribution_factory,
        )

        distributions, lengths = process.generate_segments()

        assert sum(lengths) == total_length
        assert len(distributions) == len(lengths)
        assert all(isinstance(length, int) for length in lengths)
        assert all(length >= 0 for length in lengths)

        for dist in distributions:
            assert isinstance(dist, MockDistribution)
            assert dist._mean == mean_value
