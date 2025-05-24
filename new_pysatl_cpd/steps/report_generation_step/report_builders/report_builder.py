from abc import ABC, abstractmethod
from typing import Any, Optional


class ReportBuilder(ABC):
    """Abstract base class for building and transforming report data.

    Provides a complete report building for:
    1. Generating raw report data (implemented by subclasses)
    2. Filtering and renaming fields
    3. Delivering formatted results

    :param builder_result_fields: Fields to include in final output. Can be:
        - A set of field names to keep (no renaming)
        - A dict mapping {source_name: target_name} for renaming

    :ivar _builder_result_fields: Configured output field specifications

    .. rubric:: Implementation Workflow

    1. Subclass implements :meth:`_build` to generate raw data
    2. Base class handles field filtering/renaming
    3. Final results delivered via :meth:`__call__`
    """

    def __init__(self, builder_result_fields: Optional[set[str] | dict[str, str]] = None):
        self._builder_result_fields = builder_result_fields if builder_result_fields else set()

    def _filter_and_rename(self, report_builder_result: dict[str, float]) -> dict[str, float]:
        """Apply field selection and optional renaming to report data.

        :param report_builder_result: Complete generated report data
        :return: Processed report with only specified fields
        :raises ValueError: If any configured field is missing from input

        .. note::
            - With set configuration: keeps original field names
            - With dict configuration: renames fields according to mapping
            - Always validates all configured fields exist in input
        """
        filtered_result = dict()
        for key in self._builder_result_fields:
            if key not in report_builder_result:
                raise ValueError(f"No {key} in report_builder_result")

            if isinstance(self._builder_result_fields, dict):
                filtered_result[self._builder_result_fields[key]] = report_builder_result[key]
            else:
                filtered_result[key] = report_builder_result[key]

        return filtered_result

    @abstractmethod
    def _build(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Generate report data (for example some metrics) (implemented by subclasses).

        :param kwargs: Input data for report generation
        :return: Complete dictionary of generated metrics

        .. rubric:: Implementation Guide

        Subclasses should:
        - Perform all required calculations
        - Return complete results (before filtering)
        - Document their specific input requirements

        Example::

            def _build(self, *args, **kwargs) -> dict[str, float]:
                values = kwargs["measurements"]
                return {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "range": max(values) - min(values),
                }
        """
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Execute the full report building.

        :param args: Positional arguments forwarded to _build
        :param kwargs: Keyword arguments forwarded to _build
        :return: Final processed report data
        :raises ValueError: If field validation fails

        .. note::
            - Calls _build() to generate raw data
            - Applies field filtering/renaming
            - Returns final processed results
        """
        result = self._build(*args, **kwargs)
        filtered_result = self._filter_and_rename(result)
        return filtered_result
