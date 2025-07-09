"""
Module contains project typings
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import numpy as np

StorageNames = set[str]
StorageNamesRename = dict[str, str]
StorageValues = int | float | str | list[int | str | float] | tuple[int | str | float] | np.ndarray | dict[Any, Any]
