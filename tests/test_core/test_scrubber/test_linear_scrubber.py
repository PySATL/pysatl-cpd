import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider
from pysatl_cpd.core.scrubber.linear import LinearScrubber


class TestLinearScrubber:
    @settings(max_examples=1000)
    @given(st.integers(0, 100), st.integers(1, 100), st.floats(0.01, 1))
    def test_get_windows(self, data_length, window_length, shift_factor):
        data = [float(i) for i in range(data_length)]
        scrubber = LinearScrubber(ListUnivariateProvider(data), window_length, shift_factor)
        cur_index = 0
        for window in iter(scrubber):
            assert len(window.values) == len(window.indices)
            assert np.array_equal(window.values, np.fromiter(data[cur_index : cur_index + window_length], np.float64))
            cur_index += max(1, int(window_length * shift_factor))

    @settings(max_examples=1000)
    @given(st.integers(0, 100), st.integers(1, 100), st.floats(0.01, 1), st.integers(0, 100))
    def test_restart(self, data_length, window_length, shift_factor, window_start):
        data = [i for i in range(data_length)]
        scrubber = LinearScrubber(ListUnivariateProvider(data), window_length, shift_factor)
        fst = list(scrubber)
        snd = list(scrubber)
        assert len(fst) == len(snd)
        assert all(
            map(lambda w: w[0].indices == w[1].indices and np.array_equal(w[0].values, w[1].values), zip(fst, snd))
        )
