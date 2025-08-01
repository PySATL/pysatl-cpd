"""
Module for generating datasets with specified distributions using different backends (e.g. SciPy).
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .config_parser import ConfigParser
from .distributions import Distribution
from .saver import DatasetSaver


class Generators(Enum):
    SCIPY = "scipy"

    def __str__(self) -> str:
        return self.value


class DatasetGenerator(ABC):
    """
    An interface for dataset generators using different backends (e.g. SciPy or Numpy)
    to create a sample with a given distributions and lengths.
    """

    @abstractmethod
    def generate_sample(self, distributions: list[Distribution], lengths: list[int]) -> npt.NDArray[np.float64]:
        """
        Creates a sample consists of subsamples with given `distributions` and `lengths`.

        :param distributions: List of distributions for subsamples.
        :param lengths: List of subsamples lengths.
        :return: Created sample.
        """
        raise NotImplementedError()

    @staticmethod
    def get_generator(generator_backend: Generators) -> "DatasetGenerator":
        match generator_backend:
            case Generators.SCIPY:
                return ScipyDatasetGenerator()
            case _:
                raise ValueError("Unknown generator")

    def generate_datasets(
        self, config_path: Path, saver: DatasetSaver | None = None
    ) -> dict[str, tuple[npt.NDArray[np.float64], list[int]]]:
        """Generate pairs of dataset and change points by config file

        :param config_path: path to config file
        :param saver: saver of saving files (if saver is None, then the data does not need to be saved),
         defaults to None

        :return: dictionary with names and pairs of dataset and change points
        """
        config_parser: ConfigParser = ConfigParser(config_path)

        datasets = dict()

        for descr in config_parser:
            sample = self.generate_sample(descr.distributions, descr.length)
            current_point = 0
            change_points = []
            for length in descr.length[:-1]:
                current_point += length
                change_points.append(current_point)
            datasets[descr.name] = (sample, change_points)
            if saver:
                saver.save_sample(sample, descr)
        return datasets


class ScipyDatasetGenerator(DatasetGenerator):
    """
    Dataset generator using SciPy to create samples.
    """

    def generate_sample(self, distributions: list[Distribution], lengths: list[int]) -> npt.NDArray[np.float64]:
        return np.concatenate(
            [distribution.scipy_sample(length) for distribution, length in zip(distributions, lengths)]
        )
