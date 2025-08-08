"""Generates time series segments, each segment characterized by a length and distribution."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from numpy.random import Generator
import scipy.stats as sc

from pysatl_cpd.generator.distributions import (
    Distribution,
)


class ChangepointProcess(ABC):
    """Abstract base class for a changepoint process.

    This class defines the interface for generating a sequence of data segments
    separated by changepoints and characterized with its own distribution.

    .. rubric:: Implementation Requirements

    Subclasses must:

    1. Implement the :meth:`generate_segments` method to define the generation logic.
    """

    @abstractmethod
    def generate_segments(self) -> tuple[list[Distribution], list[int]]:
        """Generates the distributions and lengths of segments.

        This method must be implemented by subclasses to define the specific
        logic for generating the sequence of data distributions and their
        corresponding lengths that make up the time series.

        :return: A tuple containing two lists:
                 - A list of Distribution objects for each segment.
                 - A list of integer lengths for each corresponding segment.
        """


class PoissonChangepointProcess(ChangepointProcess):
    """Generates segments where changepoints occur based on a Poisson process.

    In this model, the lengths of the segments between changepoints are sampled
    from an exponential distribution, which is a characteristic of a Poisson process.

    :param total_length: The total desired length of the time series.
    :param cp_intensity_per_point: The probability of a changepoint at any given point,
                                   used to determine the average segment length.
    :param mean_sampler: A Distribution object used to sample the mean for each new segment.
    :param distribution_factory: A callable that takes a mean and returns a configured
                                 Distribution object for a segment.
    :param random_state: An integer required to reproduce the behavior of the class.

    :ivar _total_length: The total length of the time series to be generated.
    :ivar _avg_segment_length: The average segment length, calculated as the inverse of
                               cp_intensity_per_point.
    :ivar _mean_sampler: The sampler for generating segment means.
    :ivar _distribution_factory: The factory for creating segment distributions.
    :ivar rng: Generator object for generating random variates.
    """

    def __init__(
        self,
        total_length: int,
        cp_intensity_per_point: float,
        mean_sampler: Distribution,
        distribution_factory: Callable[[float], Distribution],
        random_state: int = 42
    ):
        """Initializes the PoissonChangepointProcess.

        :raises ValueError: If total_length or cp_intensity_per_point are not positive.
        """

        if total_length <= 0 or cp_intensity_per_point <= 0:
            raise ValueError("Length and intensity must be positive")

        self._total_length = total_length
        self._avg_segment_length = 1.0 / cp_intensity_per_point
        self._mean_sampler = mean_sampler
        self._distribution_factory = distribution_factory
        self.rng: Generator = np.random.default_rng(random_state)

    def generate_segments(self) -> tuple[list[Distribution], list[int]]:
        distributions: list[Distribution] = []
        lengths: list[int] = []

        exp_dist = sc.expon(scale=self._avg_segment_length)

        current_length = 0
        while current_length < self._total_length:

            remaining_length = self._total_length - current_length
            proposed_len = max(1, round(exp_dist.rvs(1, random_state=self.rng)[0]))
            segment_len = min(proposed_len, remaining_length)
            lengths.append(segment_len)

            mean_for_segment = self._mean_sampler.scipy_sample(1)[0]
            dist_for_segment = self._distribution_factory(mean_for_segment)
            distributions.append(dist_for_segment)

            current_length += segment_len

        return distributions, lengths
