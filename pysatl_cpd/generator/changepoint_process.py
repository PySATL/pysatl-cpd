from abc import ABC, abstractmethod
from typing import Callable

import scipy.stats as sc

from pysatl_cpd.generator.distributions import (
    Distribution,
)


class ChangepointProcess(ABC):
    """
    Abstract class for processes that generate change points (via segment lengths).
    """

    @abstractmethod
    def generate_segments(self) -> tuple[list[Distribution], list[int]]:
        """
        Generates a list of distributions and their corresponding segment lengths.

        :return: A tuple of a list of distributions and a list of lengths.
        """
        raise NotImplementedError


class PoissonChangepointProcess(ChangepointProcess):
    def __init__(
        self,
        total_length: int,
        cp_intensity: float,
        mean_sampler: Distribution,
        distribution_factory: Callable[[float], Distribution],
    ):
        """
        :param total_length: The total desired length of the final dataset.
        :param avg_segment_length: Average expected segment length (parameter `lambda` for Poisson).
        :param mean_sampler: The distribution from which the average values for each segment will be generated
        :param distribution_factory: Factory function that generates a Distribution object
                                    for each segment based on the means returned by mean_sampler
        """
        if total_length <= 0 or cp_intensity <= 0:
            raise ValueError("Length and intensity must be positive")

        self._total_length = total_length
        self._avg_segment_length = 1.0 / cp_intensity
        self._mean_sampler = mean_sampler
        self._distribution_factory = distribution_factory

    def generate_segments(self) -> tuple[list[Distribution], list[int]]:
        distributions: list[Distribution] = []
        lengths: list[int] = []

        exp_dist = sc.expon(scale=self._avg_segment_length)

        current_length = 0
        while current_length < self._total_length:
            segment_len = int(round(exp_dist.rvs(1)[0]))

            if current_length + segment_len > self._total_length:
                segment_len = self._total_length - current_length

            lengths.append(segment_len)

            mean_for_segment = self._mean_sampler.scipy_sample(1)[0]

            dist_for_segment = self._distribution_factory(mean_for_segment)
            distributions.append(dist_for_segment)

            current_length += segment_len

        return distributions, lengths
