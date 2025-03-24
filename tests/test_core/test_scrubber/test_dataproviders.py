__author__ = "Vladimir Kutuev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from hypothesis import given, strategies

from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider


class TestDataProviders:
    @given(strategies.lists(strategies.floats(allow_nan=False), min_size=0, max_size=100))
    def test_list_univariate(self, data: list[float]):
        provider = ListUnivariateProvider(data)
        provided_data = list(provider.__iter__())
        assert len(data) == len(provided_data)
        assert all(map(lambda t: t[0] == t[1], zip(data, provided_data)))
