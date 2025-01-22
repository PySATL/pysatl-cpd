"""
Module for implementation of CPD algorithm based on knn classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2025 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import typing as tp

import yaml

from CPDShell.generator.generator import ScipyDatasetGenerator
from CPDShell.generator.saver import DatasetSaver


class DistributionType(Enum):
    normal = 1
    exponential = 2
    uniform = 3
    weibull = 4
    beta = 5


@dataclass
class Distribution:
    type: DistributionType
    parameters: dict[str, float]
    length: int


DistributionComposition = list[Distribution]


class VerboseSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


class DistributionGenerator:
    @staticmethod
    def generate(distributions: list[DistributionComposition], sample_count: int, dest_path: Path):
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        distributions_info = DistributionGenerator.__generate_configs(distributions, sample_count, dest_path)
        DistributionGenerator.__generate_experiment_description(distributions_info, dest_path)
        DistributionGenerator.__generate_dataset(distributions_info, dest_path)

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
    def __generate_configs(
        distributions: list[DistributionComposition], sample_count: int, dest_path: Path
    ) -> list[tuple[str, int]]:
        generated_distributions_info = []

        for i in range(len(distributions)):
            distribution_comp = distributions[i]

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
