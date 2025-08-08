import hypothesis.strategies as st
import numpy as np
import numpy.typing as npt
import pytest
import scipy.stats as sc
from hypothesis import given, settings
from numpy.random import Generator

from pysatl_cpd.generator import Distribution, NormalDistribution, PoissonChangepointProcess, ScipyDatasetGenerator


class MockDistribution(Distribution):
    """A mock Distribution for testing purposes."""

    def __init__(self, mean: float) -> None:
        self._mean = mean

    def scipy_sample(self, length: int) -> npt.NDArray[np.float64]:
        """Returns a simple array of the mean value."""

        return np.full(length, self._mean, dtype=np.float64)

    @property
    def name(self) -> str:
        return "mock"

    @property
    def params(self) -> dict[str, str]:
        return {"mean": str(self._mean)}

    @classmethod
    def from_params(cls, params: dict[str, str]) -> "Distribution":
        """Mock does not support instantiation from params."""

        raise NotImplementedError


def mock_distribution_factory(mean: float) -> Distribution:
    """A factory function that returns a MockDistribution instance."""

    return MockDistribution(mean)


class TestPoissonChangepointProcess:
    """
    Tests for the PoissonChangepointProcess class.
    """

    def test_initialization_success(self):
        """
        Tests successful initialization of PoissonChangepointProcess
        and verifies that attributes are set correctly.
        """

        total_length = 1000
        cp_intensity = 0.01
        mean_sampler = MockDistribution(mean=10)
        random_state = 42

        process = PoissonChangepointProcess(
            total_length=total_length,
            cp_intensity_per_point=cp_intensity,
            mean_sampler=mean_sampler,
            distribution_factory=mock_distribution_factory,
            random_state=random_state,
        )

        assert process._total_length == total_length
        assert process._avg_segment_length == 1.0 / cp_intensity
        assert process._mean_sampler is mean_sampler
        assert process._distribution_factory is mock_distribution_factory
        assert isinstance(process.rng, Generator)

    @pytest.mark.parametrize(
        "length, intensity, error_msg",
        [
            (-100, 0.1, "Length and intensity must be positive"),
            (100, -0.1, "Length and intensity must be positive"),
            (0, 0.1, "Length and intensity must be positive"),
            (100, 0, "Length and intensity must be positive"),
        ],
    )
    def test_initialization_raises_value_error_for_non_positive_params(
        self, length: int, intensity: float, error_msg: str
    ):
        """
        Tests that ValueError is raised for non-positive total_length or
        cp_intensity_per_point.
        """

        with pytest.raises(ValueError, match=error_msg):
            PoissonChangepointProcess(
                total_length=length,
                cp_intensity_per_point=intensity,
                mean_sampler=MockDistribution(mean=0),
                distribution_factory=mock_distribution_factory,
            )

    @given(
        total_length=st.integers(min_value=1, max_value=5000),
        cp_intensity=st.floats(min_value=0.001, max_value=0.8),
        random_state=st.integers(min_value=0, max_value=100000),
    )
    @settings(max_examples=50, deadline=None)
    def test_generate_segments_output_properties_with_hypothesis(
        self, total_length: int, cp_intensity: float, random_state: int
    ):
        """
        Tests the basic output properties of generate_segments using Hypothesis.
        - The sum of lengths equals total_length.
        - The number of distributions matches the number of segments.
        - All lengths are positive.
        - All returned distributions are of the correct type.
        """

        process = PoissonChangepointProcess(
            total_length=total_length,
            cp_intensity_per_point=cp_intensity,
            mean_sampler=MockDistribution(mean=1.0),
            distribution_factory=mock_distribution_factory,
            random_state=random_state,
        )

        distributions, lengths = process.generate_segments()

        assert isinstance(distributions, list)
        assert isinstance(lengths, list)
        assert all(isinstance(d, Distribution) for d in distributions)
        assert all(isinstance(length, int) and length > 0 for length in lengths)
        assert len(distributions) == len(lengths)
        assert sum(lengths) == total_length

    def test_generate_segments_reproducibility(self):
        """
        Tests that using the same random_state produces the exact same segments.
        """

        process1 = PoissonChangepointProcess(
            total_length=500,
            cp_intensity_per_point=0.02,
            mean_sampler=MockDistribution(mean=5),
            distribution_factory=mock_distribution_factory,
            random_state=123,
        )
        process2 = PoissonChangepointProcess(
            total_length=500,
            cp_intensity_per_point=0.02,
            mean_sampler=MockDistribution(mean=5),
            distribution_factory=mock_distribution_factory,
            random_state=123,
        )

        distributions1, lengths1 = process1.generate_segments()
        distributions2, lengths2 = process2.generate_segments()

        assert lengths1 == lengths2
        means1 = [d.params["mean"] for d in distributions1]
        means2 = [d.params["mean"] for d in distributions2]
        assert means1 == means2

    def test_segment_lengths_follow_exponential_distribution_ks_test(self):
        """
        Performs a Kolmogorov-Smirnov test to verify that segment lengths
        are exponentially distributed, which is a key property of a Poisson process.
        The null hypothesis (H0) is that the segment lengths are drawn from an
        exponential distribution. A high p-value (> 0.05) means we cannot
        reject H0.
        """

        total_length = 50000  # A large sample size for a meaningful statistical test
        cp_intensity = 0.01  # Average segment length = 1/0.01 = 100
        avg_segment_length = 1.0 / cp_intensity

        process = PoissonChangepointProcess(
            total_length=total_length,
            cp_intensity_per_point=cp_intensity,
            mean_sampler=MockDistribution(mean=0),
            distribution_factory=mock_distribution_factory,
            random_state=42,
        )

        _, lengths = process.generate_segments()

        # The last segment's length is truncated to fit total_length,
        # so we exclude it from the statistical test to avoid bias.
        observed_lengths = lengths[:-1]
        min_num_of_lengths = 30

        if len(observed_lengths) < min_num_of_lengths:
            pytest.skip("Not enough segments generated for a reliable K-S test.")

        # H0: The observed lengths are from an exponential distribution.
        min_p_value = 0.05
        ks_statistic, p_value = sc.kstest(observed_lengths, "expon", args=(0, avg_segment_length))

        assert p_value > min_p_value

    def test_integration_with_scipy_dataset_generator(self):
        """
        Tests that the output of PoissonChangepointProcess works correctly
        as input for ScipyDatasetGenerator to generate a final sample.
        This test uses real distributions instead of mocks.
        """

        total_length = 2000
        mean_sampler = NormalDistribution(mean=0, var=1.0)

        def normal_distribution_factory(mean: float) -> Distribution:
            return NormalDistribution(mean=mean, var=abs(mean) + 0.1)

        process = PoissonChangepointProcess(
            total_length=total_length,
            cp_intensity_per_point=0.05,
            mean_sampler=mean_sampler,
            distribution_factory=normal_distribution_factory,
            random_state=99,
        )
        distributions, lengths = process.generate_segments()

        scipy_generator = ScipyDatasetGenerator()

        final_sample = scipy_generator.generate_sample(distributions, lengths)

        assert isinstance(final_sample, np.ndarray)
        assert final_sample.dtype == np.float64
        assert len(final_sample) == total_length
