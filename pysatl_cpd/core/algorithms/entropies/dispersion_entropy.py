from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class DispersionEntropyAlgorithm(OnlineAlgorithm):
    def __init__(
        self,
        window_size: int = 100,
        embedding_dim: int = 3,
        num_classes: int = 6,
        time_delay: int = 1,
        threshold: float = 0.2,
        normalize: bool = True,
    ):
        self._window_size = window_size
        self._embedding_dim = embedding_dim
        self._num_classes = num_classes
        self._time_delay = time_delay
        self._threshold = threshold
        self._normalize = normalize

        if num_classes**embedding_dim >= window_size:
            raise ValueError(
                f"c^w ({num_classes}^{embedding_dim} = {num_classes**embedding_dim}) "
                f"should be less than window_size ({window_size})"
            )

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
        self._buffer.append(observation)
        self._position += 1
        v = 10
        v1 = 2

        min_required = self._window_size + (self._embedding_dim - 1) * self._time_delay
        if len(self._buffer) < min_required:
            return

        current_window = np.array(list(self._buffer)[-self._window_size :])
        current_entropy = self._calculate_dispersion_entropy(current_window)

        if np.isinf(current_entropy) or np.isnan(current_entropy):
            current_entropy = 0.0

        self._entropy_values.append(current_entropy)

        if len(self._entropy_values) >= v1:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])

            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

        if len(self._entropy_values) >= v1 + 3:
            recent_entropies = self._entropy_values[-5:]
            entropy_variance = np.var(recent_entropies)
            np.mean(recent_entropies)

            if entropy_variance > self._threshold:
                self._last_change_point = self._position - self._window_size // 4

        if len(self._entropy_values) >= v:
            window1 = self._entropy_values[-10:-5]
            window2 = self._entropy_values[-5:]
            mean1 = np.mean(window1)
            mean2 = np.mean(window2)

            if abs(mean2 - mean1) > self._threshold * 1.5:
                self._last_change_point = self._position - 2

    def _calculate_dispersion_entropy(self, time_series: npt.NDArray[np.float64]) -> float:
        N = len(time_series)
        if self._embedding_dim > N:
            return 0.0

        mu = np.mean(time_series)
        sigma = np.std(time_series)
        if sigma == 0:
            return 0.0

        y_series = norm.cdf(time_series, loc=mu, scale=sigma)
        z_series = self._discretize_to_classes(y_series)
        patterns = self._create_dispersion_patterns(z_series)
        pattern_probs = self._calculate_pattern_probabilities(patterns)
        de_value = self._calculate_shannon_entropy(pattern_probs)

        if self._normalize:
            max_entropy = np.log(self._num_classes**self._embedding_dim)
            if max_entropy > 0:
                de_value = de_value / max_entropy

        return float(de_value)

    def _discretize_to_classes(self, y_series: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        rc_values = self._num_classes * y_series + 0.5
        z_series = np.round(rc_values).astype(np.int32)
        z_series = np.clip(z_series, 1, self._num_classes)
        return z_series

    def _create_dispersion_patterns(self, z_series: npt.NDArray[np.int32]) -> list[tuple]:
        N = len(z_series)
        patterns = []

        for i in range(N - (self._embedding_dim - 1) * self._time_delay):
            pattern = []
            for j in range(self._embedding_dim):
                idx = i + j * self._time_delay
                if idx < N:
                    pattern.append(z_series[idx])
            if len(pattern) == self._embedding_dim:
                patterns.append(tuple(pattern))

        return patterns

    def _calculate_pattern_probabilities(self, patterns: list[tuple]) -> dict[tuple, float]:
        if not patterns:
            return {}

        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        total_patterns = len(patterns)
        pattern_probs = {}
        for pattern, count in pattern_counts.items():
            pattern_probs[pattern] = count / total_patterns

        return pattern_probs

    def _calculate_shannon_entropy(self, pattern_probs: dict[tuple, float]) -> float:
        if not pattern_probs:
            return 0.0

        entropy = 0.0
        for prob in pattern_probs.values():
            if prob > 0:
                entropy -= prob * np.log(prob)

        return entropy

    def get_entropy_history(self) -> list[float]:
        return self._entropy_values.copy()

    def get_current_parameters(self) -> dict:
        return {
            "window_size": self._window_size,
            "embedding_dim": self._embedding_dim,
            "num_classes": self._num_classes,
            "time_delay": self._time_delay,
            "threshold": self._threshold,
            "normalize": self._normalize,
            "max_patterns": self._num_classes**self._embedding_dim,
        }

    def set_parameters(
        self,
        embedding_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        time_delay: Optional[int] = None,
        threshold: Optional[float] = None,
        normalize: Optional[bool] = None,
    ) -> None:
        if embedding_dim is not None:
            self._embedding_dim = embedding_dim
        if num_classes is not None:
            self._num_classes = num_classes
        if time_delay is not None:
            self._time_delay = time_delay
        if threshold is not None:
            self._threshold = threshold
        if normalize is not None:
            self._normalize = normalize

        if self._num_classes**self._embedding_dim >= self._window_size:
            raise ValueError(
                f"c^w ({self._num_classes}^{self._embedding_dim}) should be less than window_size ({self._window_size})"
            )

    def get_pattern_distribution(self) -> dict[tuple, int]:
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.array(list(self._buffer)[-self._window_size :])
        mu = np.mean(current_window)
        sigma = np.std(current_window)
        if sigma == 0:
            return {}

        y_series = norm.cdf(current_window, loc=mu, scale=sigma)
        z_series = self._discretize_to_classes(y_series)
        patterns = self._create_dispersion_patterns(z_series)

        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        return pattern_counts

    def analyze_complexity(self) -> dict:
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.array(list(self._buffer)[-self._window_size :])
        current_entropy = self._calculate_dispersion_entropy(current_window)

        pattern_dist = self.get_pattern_distribution()
        unique_patterns = len(pattern_dist)
        max_possible_patterns = self._num_classes**self._embedding_dim

        pattern_diversity = unique_patterns / max_possible_patterns if max_possible_patterns > 0 else 0

        max_entropy = np.log(max_possible_patterns) if max_possible_patterns > 0 else 1
        relative_entropy = current_entropy / max_entropy if max_entropy > 0 else 0

        return {
            "dispersion_entropy": current_entropy,
            "normalized_entropy": relative_entropy,
            "unique_patterns": unique_patterns,
            "max_possible_patterns": max_possible_patterns,
            "pattern_diversity": pattern_diversity,
            "window_std": np.std(current_window),
            "window_mean": np.mean(current_window),
        }

    def reset(self) -> None:
        self._buffer.clear()
        self._entropy_values.clear()
        self._position = 0
        self._last_change_point = None
