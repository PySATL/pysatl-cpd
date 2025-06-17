from collections import Counter, deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class RenyiEntropyAlgorithm(OnlineAlgorithm):
    def __init__(
        self,
        window_size: int = 100,
        alpha: float = 0.5,
        bins: int = 10,
        threshold: float = 0.3,
    ):
        if alpha <= 0 or alpha == 1:
            raise ValueError("Alpha must be positive and not equal to 1")

        self._window_size = window_size
        self._alpha = alpha
        self._bins = bins
        self._threshold = threshold

        self._buffer: deque[float] = deque(maxlen=window_size * 2)
        self._entropy_values: list[float] = []
        self._position: int = 0
        self._last_change_point: Optional[int] = None
        self._global_min: Optional[float] = None
        self._global_max: Optional[float] = None

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        if isinstance(observation, np.ndarray):
            for obs in observation:
                self._process_single_observation(float(obs))
        else:
            self._process_single_observation(float(observation))

        return self._last_change_point is not None

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        change_detected = self.detect(observation)

        if change_detected:
            change_point = self._last_change_point
            self._last_change_point = None
            return change_point

        return None

    def _process_single_observation(self, observation: float) -> None:
        v = 2
        self._buffer.append(observation)
        self._position += 1

        if self._global_min is None or observation < self._global_min:
            self._global_min = observation
        if self._global_max is None or observation > self._global_max:
            self._global_max = observation

        if len(self._buffer) < self._window_size:
            return

        current_window = np.array(list(self._buffer)[-self._window_size :])
        current_entropy = self._calculate_renyi_entropy(current_window)
        self._entropy_values.append(current_entropy)

        if len(self._entropy_values) >= v:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])
            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

        if len(self._entropy_values) >= v + 3:
            recent_entropies = self._entropy_values[-5:]
            entropy_variance = np.var(recent_entropies)
            if entropy_variance > self._threshold * 2:
                self._last_change_point = self._position - self._window_size // 4

    def _calculate_renyi_entropy(self, time_series: npt.NDArray[np.float64]) -> float:
        if len(time_series) == 0:
            return 0.0

        probabilities = self._compute_probabilities(time_series)
        if len(probabilities) == 0 or all(p == 0 for p in probabilities):
            return 0.0

        if self._alpha == 0:
            non_zero_count = sum(1 for p in probabilities if p > 0)
            return float(np.log(non_zero_count))
        elif np.isinf(self._alpha):
            max_prob = max(probabilities)
            return float(-np.log(max_prob)) if max_prob > 0 else 0.0
        else:
            power_sum = sum(p**self._alpha for p in probabilities if p > 0)
            if power_sum <= 0:
                return 0.0
            renyi_entropy = (1 / (1 - self._alpha)) * np.log(power_sum)
            return float(renyi_entropy)

    def _compute_probabilities(self, time_series: npt.NDArray[np.float64]) -> list[float]:
        if self._global_min is None or self._global_max is None:
            return []

        if self._global_max == self._global_min:
            return [1.0]

        bin_edges = np.linspace(self._global_min, self._global_max, self._bins + 1)
        digitized = np.digitize(time_series, bin_edges)
        bin_counts = Counter(digitized)
        total_count = len(time_series)

        probabilities = []
        for i in range(1, len(bin_edges)):
            key = np.int64(i)
            count = bin_counts.get(key, 0)
            prob = count / total_count if total_count > 0 else 0.0
            probabilities.append(prob)

        return probabilities

    def get_entropy_history(self) -> list[float]:
        return self._entropy_values.copy()

    def reset(self) -> None:
        self._buffer.clear()
        self._entropy_values.clear()
        self._position = 0
        self._last_change_point = None
        self._global_min = None
        self._global_max = None
