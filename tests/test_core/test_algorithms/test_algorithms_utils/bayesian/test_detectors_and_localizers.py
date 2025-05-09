import numpy as np
import pytest

from pysatl_cpd.core.algorithms.bayesian.detectors.drop import DropDetector
from pysatl_cpd.core.algorithms.bayesian.detectors.threshold import ThresholdDetector
from pysatl_cpd.core.algorithms.bayesian.localizers.argmax import ArgmaxLocalizer


@pytest.fixture(
    params=[
        pytest.param((ThresholdDetector, 0.8), id="Threshold"),
        pytest.param((DropDetector, 0.1), id="Drop"),
    ],
    scope="function",
)
def detector(request):
    cls, threshold = request.param
    return cls(threshold)


def generate_test_data(has_cp=True):
    run_length_probs = np.full(100, 1.0)
    run_length_probs[-1] = 50 if has_cp else 500

    return run_length_probs / run_length_probs.sum()


class TestDetectors:
    def test_detection(self, detector):
        before_cp = generate_test_data(has_cp=False)
        after_cp = generate_test_data(has_cp=True)
        print(before_cp[-1], after_cp[-1])
        assert not detector.detect(before_cp), (
            "Detector should not react in case of stable high probability of max run length"
        )
        assert detector.detect(after_cp), (
            "Detector should react in case of significant abrupt drop of probability of max run length"
        )

    def test_clear(self, detector):
        cp_data = generate_test_data(has_cp=True)

        first_result = detector.detect(cp_data)
        detector.clear()

        second_result = detector.detect(cp_data)

        assert first_result == second_result, "A state was not cleared correctly"


class TestDropDetectorSpecific:
    def test_gradual_change(self):
        detector = DropDetector(0.1)
        run_lengths = np.full(100, 1.0)

        for value in range(500, 490, -1):
            run_lengths[-1] = value
            data = run_lengths / run_lengths.sum()
            assert not detector.detect(data), "Drop detector should not react on a gradual probability decrease"


class TestArgmaxLocalizer:
    def test_localization(self):
        change_point = 5
        run_lengths = np.full(11, 0.05)
        run_lengths[change_point] = 0.5
        localizer = ArgmaxLocalizer()
        result = localizer.localize(run_lengths)
        assert result == change_point, f"Expected change at {change_point}, got {result}"
