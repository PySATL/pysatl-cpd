import numpy as np
import pytest

from pysatl_cpd.core.algorithms.bayesian.hazards.constant import ConstantHazard


class TestConstantHazard:
    @pytest.mark.parametrize("hazard_rate,max_run_length", [(1.1, 50), (10, 100), (200, 250), (500.325251, 500)])
    def test_constant_hazard_for_constants(self, hazard_rate, max_run_length):
        constant_hazard = ConstantHazard(hazard_rate)
        run_lengths = np.arange(max_run_length, dtype=np.intp)
        hazard_probs = constant_hazard.hazard(run_lengths)
        assert hazard_probs.shape[0] == max_run_length, (
            f"Expected {max_run_length} probabilities, got {hazard_probs.shape[0]}"
        )
        assert np.all(hazard_probs == 1 / hazard_rate), f"Hazard probabilities must be {1 / hazard_rate}"
