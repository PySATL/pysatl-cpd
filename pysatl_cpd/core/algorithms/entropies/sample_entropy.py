from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class SampleEntropyAlgorithm(OnlineAlgorithm):
    def __init__(
        self,
        window_size: int = 100,
        m: int = 2,
        r: Optional[float] = None,
        r_factor: float = 0.2,
        threshold: float = 0.5,
    ):
        self._window_size = window_size
        self._m = m
        self._r = r
        self._r_factor = r_factor
        self._threshold = threshold

        self._buffer: deque[float] = deque(maxlen=window_size * 2)
        self._entropy_values: list[float] = []
        self._position: int = 0
        self._last_change_point: Optional[int] = None

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
        v = 3
        self._buffer.append(observation)
        self._position += 1

        min_required = self._window_size + self._m
        if len(self._buffer) < min_required:
            return

        current_window = np.array(list(self._buffer)[-self._window_size :])
        current_entropy = self._calculate_sample_entropy(current_window)

        if np.isinf(current_entropy) or np.isnan(current_entropy):
            current_entropy = float("inf")

        self._entropy_values.append(current_entropy)

        if len(self._entropy_values) >= v - 1:
            prev_entropy = self._entropy_values[-2]
            curr_entropy = self._entropy_values[-1]

            if np.isinf(prev_entropy) and np.isinf(curr_entropy):
                entropy_diff = 0.0
            elif np.isinf(prev_entropy):
                entropy_diff = float("inf") if curr_entropy != 0 else 0.0
            elif np.isinf(curr_entropy):
                entropy_diff = float("inf") if prev_entropy != 0 else 0.0
            else:
                entropy_diff = abs(curr_entropy - prev_entropy)

            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

        if len(self._entropy_values) >= v + 2:
            recent_entropies = [e for e in self._entropy_values[-5:] if not np.isinf(e)]
            if len(recent_entropies) >= v:
                entropy_variance = np.var(recent_entropies)
                mean_entropy = np.mean(recent_entropies)

                if mean_entropy > 0 and entropy_variance / mean_entropy > self._threshold:
                    self._last_change_point = self._position - self._window_size // 4

    def _calculate_sample_entropy(self, time_series: npt.NDArray[np.float64]) -> float:
        N = len(time_series)
        if self._m + 1 > N:
            return float("inf")

        r = self._r
        if r is None:
            std_dev = np.std(time_series)
            if std_dev == 0:
                return float("inf")
            r = self._r_factor * std_dev

        B = self._count_matches(time_series, self._m, r)
        A = self._count_matches(time_series, self._m + 1, r)

        if B == 0:
            return float("inf")
        if A == 0:
            return float("inf")

        sample_entropy = -np.log(A / B)
        return float(sample_entropy)

    def _count_matches(self, time_series: npt.NDArray[np.float64], m: int, r: float) -> int:
        N = len(time_series)
        matches = 0

        for i in range(N - m + 1):
            xi = time_series[i : i + m]
            for j in range(i + 1, N - m + 1):
                xj = time_series[j : j + m]
                distance = self._chebyshev_distance(xi, xj)
                if distance < r:
                    matches += 1

        return matches

    def _chebyshev_distance(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        return float(np.max(np.abs(x - y)))

    def _euclidean_distance(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        return float(np.sqrt(np.sum((x - y) ** 2)))

    def get_entropy_history(self) -> list[float]:
        return self._entropy_values.copy()

    def get_current_r(self) -> Optional[float]:
        if self._r is not None:
            return self._r

        if len(self._buffer) > 0:
            current_window = np.array(list(self._buffer)[-self._window_size :])
            std_dev = np.std(current_window)
            return self._r_factor * std_dev if std_dev > 0 else None

        return None

    def reset(self) -> None:
        self._buffer.clear()
        self._entropy_values.clear()
        self._position = 0
        self._last_change_point = None
