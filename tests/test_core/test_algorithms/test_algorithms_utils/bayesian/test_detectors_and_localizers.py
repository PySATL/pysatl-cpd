import numpy as np

from pysatl_cpd.core.algorithms.bayesian.detectors.threshold import ThresholdDetector
from pysatl_cpd.core.algorithms.bayesian.localizers.argmax import ArgmaxLocalizer


class TestThresholdDetector:
    def test_detection(self):
        run_lengths = np.full(100, 0.01)
        detector = ThresholdDetector(threshold=0.04)
        assert detector.detect(run_lengths), "Change point should be detected"

    def test_clear(self):
        run_lengths = np.full(100, 0.01)
        detector = ThresholdDetector(threshold=0.04)

        result1 = detector.detect(run_lengths)
        detector.clear()
        result2 = detector.detect(run_lengths)

        assert result1 == result2, f"Results differ after clear: {result1} vs {result2}"


class TestArgmaxLocalizer:
    def test_localization(self):
        change_point = 5
        run_lengths = np.full(11, 0.05)
        run_lengths[change_point] = 0.5
        localizer = ArgmaxLocalizer()
        result = localizer.localize(run_lengths)
        assert result == change_point, f"Expected change at {change_point}, got {result}"
