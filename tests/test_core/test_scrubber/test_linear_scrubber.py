import hypothesis.strategies as st
from hypothesis import given, settings

from pysatl_cpd.core.scrubber.data_providers import ListUnivariateProvider
from pysatl_cpd.core.scrubber.linear import LinearScrubber


class TestLinearScrubber:
    @settings(max_examples=1000)
    @given(st.integers(0, 100), st.integers(1, 100), st.floats(0.01, 1))
    def test_get_windows(self, data_length, window_length, shift_factor):
        data = [i for i in range(data_length)]
        scrubber = LinearScrubber(ListUnivariateProvider(data), window_length, shift_factor)
        cur_index = 0
        for window in scrubber.__iter__():
            assert window.values == data[cur_index : cur_index + window_length]
            cur_index += max(1, int(window_length * shift_factor))

    # @settings(max_examples=1000)
    # @given(st.integers(0, 100), st.integers(0, 100), st.integers(0, 100), st.floats(0.01, 1))
    # def test_add_change_points(self, max_change_point_number, data_length, window_length, shift_factor):
    #     data = [i for i in range(data_length)]
    #     scenario = ScrubberScenario(max_change_point_number, True)
    #     scrubber = LinearScrubber(window_length, shift_factor)
    #     scrubber.scenario = scenario
    #     scrubber._data = data
    #
    #     cur_index = 0
    #     for window in scrubber.get_windows():
    #         change_points_num = random.randint(0, window_length)
    #         change_points = [random.randint(0, window_length - 1) for _ in range(change_points_num)]
    #         scrubber_cp_number = len(scrubber.change_points)
    #         scrubber.add_change_points(change_points)
    #         for point in change_points[:max_change_point_number]:
    #             assert point + cur_index in scrubber.change_points
    #         if max_change_point_number <= change_points_num:
    #             assert len(scrubber.change_points) <= scrubber_cp_number + max_change_point_number
    #         cur_index += max(1, int(window_length * shift_factor))

    @settings(max_examples=1000)
    @given(st.integers(0, 100), st.integers(1, 100), st.floats(0.01, 1), st.integers(0, 100))
    def test_restart(self, data_length, window_length, shift_factor, window_start):
        data = [i for i in range(data_length)]
        scrubber = LinearScrubber(ListUnivariateProvider(data), window_length, shift_factor)
        fst = list(scrubber)
        snd = list(scrubber)
        assert len(fst) == len(snd)
        assert all(map(lambda w: w[0].indices == w[1].indices and w[0].values == w[1].values, zip(fst, snd)))
