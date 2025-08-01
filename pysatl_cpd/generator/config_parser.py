"""
Module for parsing and validating YAML configuration files for dataset generation.
"""

__author__ = "Romanyuk Artem, Vladimir Kutuev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import os.path
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, TypeAlias

import yaml

from .dataset_description import DatasetDescriptionBuilder, SampleDescription
from .distributions import Distribution

ConfDistrParams: TypeAlias = dict[str, str]
ConfDistr: TypeAlias = dict[str, str | int | ConfDistrParams]
ConfDistrs: TypeAlias = list[ConfDistr]
ConfDataset: TypeAlias = dict[str, str | ConfDistrs]
ParsedConfig: TypeAlias = list[ConfDataset]


class ConfigParser:
    """
    Parse YAML generation config and provides an iterator over DatasetDescriptions.
    """

    NAME_FIELD: Final[str] = "name"
    DISTRIBUTION_FIELD: Final[str] = "distributions"
    DISTRIBUTION_TYPE: Final[str] = "type"
    LENGTH_FIELD: Final[str] = "length"
    PARAMETERS_FIELD: Final[str] = "parameters"

    _descriptions: list[SampleDescription]

    def __init__(self, config_path: Path):
        self.validate_config(config_path)
        with open(config_path) as cf:
            config: ParsedConfig = yaml.safe_load(cf)
            self._descriptions = self._parse_config(config)

    def __iter__(self) -> Iterator[SampleDescription]:
        return self._descriptions.__iter__()

    @staticmethod
    def _parse_config(config: ParsedConfig) -> list[SampleDescription]:
        """Parse config to list of descriptions

        :param config: config with sample description
        :return: list of descriptions"""
        descriptions: list[SampleDescription] = []
        for descr in config:
            db = DatasetDescriptionBuilder()
            match (
                descr[ConfigParser.NAME_FIELD],
                descr[ConfigParser.DISTRIBUTION_FIELD],
            ):
                case (str(name), list(distributions)):
                    pass
                case _:
                    raise ValueError("Invalid config")
            db.set_name(name)
            for distr in distributions:
                match (
                    distr[ConfigParser.DISTRIBUTION_TYPE],
                    distr[ConfigParser.LENGTH_FIELD],
                    distr[ConfigParser.PARAMETERS_FIELD],
                ):
                    case (str(name), int(length), dict(params)):
                        pass
                    case _:
                        raise ValueError("Invalid config")
                db.add_distribution(name, length, params)
            descriptions.append(db.build())
        return descriptions

    @staticmethod
    def validate_config(config_path: Path) -> None:
        """
        Reads and validates samples generation config file.
        Raise `TypeError` or `ValueError` if config is not valid.

        :param config_path: Path to configuration file.
        """
        if not os.path.exists(config_path):
            raise ValueError("Incorrect config path")
        with open(config_path) as cf:
            config = yaml.safe_load(cf)
        if not isinstance(config, list):
            raise TypeError("Config must be a list of dataset descriptions")
        for i, descr in enumerate(config):
            if not isinstance(descr, dict):
                raise TypeError(f"Description #{i} is not a dictionary")
            name = descr.get(ConfigParser.NAME_FIELD, None)
            ConfigParser._validate_description_name(i, name)
            distributions: ConfDistrs | None = descr.get(ConfigParser.DISTRIBUTION_FIELD, None)
            if distributions is None:
                raise TypeError(f"No distribution in Description #{i}")
            ConfigParser._validate_description_list_field(i, distributions, "distributions", dict)
            for distribution in descr[ConfigParser.DISTRIBUTION_FIELD]:
                distribution_type = distribution.get(ConfigParser.DISTRIBUTION_TYPE, None)
                ConfigParser._validate_description_field_type(i, distribution_type, ConfigParser.DISTRIBUTION_TYPE, str)
                distribution_length = distribution.get(ConfigParser.LENGTH_FIELD, None)
                ConfigParser._validate_description_field_type(i, distribution_length, ConfigParser.LENGTH_FIELD, int)
                distribution_parameters = distribution.get(ConfigParser.PARAMETERS_FIELD, None)
                ConfigParser._validate_description_field_type(
                    i, distribution_parameters, ConfigParser.PARAMETERS_FIELD, dict
                )
                if distribution_type is not None and distribution_parameters is not None:
                    try:
                        Distribution.from_str(distribution_type, distribution_parameters)
                    except Exception as e:
                        raise ValueError(f"Description #{i} distribution is invalid") from e
                else:
                    raise ValueError("distributions and params can not be None")

    @staticmethod
    def _validate_description_name(index: int, name: str | None) -> None:
        """Validate dataset name

        :param index: dataset index in config
        :param name: dataset name"""
        if not name:
            raise ValueError(f"Description #{index} does not contain a name")
        if not isinstance(name, str):
            raise TypeError(f"Description #{index} name is not string")

    @staticmethod
    def _validate_description_field_type(index: int, field: Any, field_name: str, field_type: type) -> None:
        """Validate field type

        :param index: dataset index in config
        :param field: field from config
        :param field_name: name of field
        :param field_type: field type"""
        if not field:
            raise ValueError(f"Description #{index} does not contain a {field_name} list for sub-samples")
        if not isinstance(field, field_type):
            raise TypeError(f"Description #{index} {field_name} is not {field_type}")

    @staticmethod
    def _validate_description_list_field(
        index: int, list_field: ConfDistrs, list_field_name: str, elements_type: type
    ) -> None:
        """Validate list from config

        :param index: dataset index in config
        :param list_field: list from config
        :param list_field_name: name of field
        :param elements_type: type of elements in list"""
        if not list_field:
            raise ValueError(f"Description #{index} does not contain a {list_field_name} list for sub-samples")
        if not isinstance(list_field, list):
            raise TypeError(f"Description #{index} {list_field_name} is not list")
        for l_idx, length in enumerate(list_field):
            if not isinstance(length, elements_type):
                raise TypeError(f"Description #{index} lengths[{l_idx}] is not {elements_type}")
