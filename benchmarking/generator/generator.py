"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import csv
import typing as tp
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from random import randint

import yaml

from pysatl_cpd.generator.generator import ScipyDatasetGenerator
from pysatl_cpd.generator.saver import DatasetSaver


class DistributionType(Enum):
    normal = 1
    exponential = 2
    uniform = 3
    weibull = 4
    beta = 5
    multivariate_normal = 6


@dataclass
class Distribution:
    type: DistributionType
    parameters: dict[str, float]
    length: int


DistributionComposition = list[Distribution]


class VerboseSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: tp.Any) -> bool:
        return True


class DistributionGenerator:
    @staticmethod
    def generate_by_config(config_path: Path, dataset_path: Path, sample_count: int) -> list[DistributionComposition]:
        with open(config_path) as stream:
            loaded_config: list[dict[str, tp.Any]] = yaml.safe_load(stream)

        distributions: list[DistributionComposition] = []

        for distr_comp_config in loaded_config:
            distr_comp: DistributionComposition = [
                Distribution(DistributionType[distr_config["type"]], distr_config["parameters"], distr_config["length"])
                for distr_config in distr_comp_config["distributions"]
            ]
            distributions.append(distr_comp)

        DistributionGenerator.generate(distributions, sample_count, dataset_path)

        return distributions

    @staticmethod
    def generate_by_frequency_rnd(distr_length: int, overall_length: int, dest_path: Path) -> None:
        default_distrs = [Distribution(DistributionType.beta, {"alpha": 1.0, "beta": 1.0}, distr_length),
                           Distribution(DistributionType.exponential, {"rate": 1.0}, distr_length),
                           Distribution(DistributionType.uniform, {"min": 0.0, "max": 1.0}, distr_length),
                           Distribution(DistributionType.weibull, {"shape": 0.5, "scale": 1.0}, distr_length),
                           Distribution(DistributionType.normal, {"mean": 0.0, "variance": 1.0}, distr_length)]

        sample_distr = []

        prev = -1
        for _ in range(0, overall_length, distr_length):
            distr_n = randint(0, 4)
            while distr_n == prev:
                distr_n = randint(0, 4)
            prev = distr_n

            sample_distr.append(default_distrs[distr_n])

        DistributionGenerator.generate([sample_distr], 1, dest_path)

    @staticmethod
    def generate(distributions: list[DistributionComposition], sample_count: int, dest_path: Path) -> None:
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        distributions_info = DistributionGenerator.__generate_configs(distributions, sample_count, dest_path)
        DistributionGenerator.__generate_experiment_description(distributions_info, dest_path)
        DistributionGenerator.__generate_dataset(distributions_info, dest_path)

    @staticmethod
    def __generate_configs(
        distributions: list[DistributionComposition], sample_count: int, dest_path: Path
    ) -> list[tuple[str, int]]:
        generated_distributions_info = []

        for i in range(len(distributions)):
            distribution_comp = distributions[i]

            if len(distribution_comp) > 4:
                name = f"{i}-{len(distribution_comp)}-{sum(map(lambda x: x.length, distribution_comp))}"
            else:
                name = f"{i}-" + "-".join(map(lambda d: d.type.name, distribution_comp))

            generated_distributions_info.append((name, sample_count))

            config = [
                {
                    "name": name,
                    "distributions": [
                        {"type": distr_conf.type.name, "length": distr_conf.length, "parameters": distr_conf.parameters}
                        for distr_conf in distribution_comp
                    ],
                }
            ]

            Path(dest_path / Path(name)).mkdir(parents=True, exist_ok=True)
            with open(dest_path / f"{name}/config.yaml", "w") as outfile:
                yaml.dump(config, outfile, default_flow_style=False, sort_keys=False, Dumper=VerboseSafeDumper)

        return generated_distributions_info

    @staticmethod
    def __generate_experiment_description(distributions_info: list[tuple[str, int]], dest_path: Path) -> None:
        with open(dest_path / "experiment_description", "w", newline="") as f:
            write = csv.writer(f)
            write.writerow(["name", "samples_num"])
            samples_description = [[d_info[0], str(d_info[1])] for d_info in distributions_info]
            write.writerows(samples_description)

    @staticmethod
    def __generate_dataset(distributions_info: list[tuple[str, int]], dest_path: Path) -> None:
        for d_info in distributions_info:
            name = d_info[0]
            sample_count = d_info[1]

            Path(dest_path / name).mkdir(parents=True, exist_ok=True)

            for sample_num in range(sample_count):
                print(f"Name: {name}. Sample num: {sample_num}")
                Path(dest_path / f"{name}/sample_{sample_num}/").mkdir(parents=True, exist_ok=True)
                saver = DatasetSaver(dest_path / f"{name}/sample_{sample_num}/", True)
                ScipyDatasetGenerator().generate_datasets(Path(dest_path / f"{name}/config.yaml"), saver)

                Path(dest_path / f"{name}/sample_{sample_num}/{name}/sample.png").unlink(missing_ok=True)
