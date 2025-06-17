from collections import deque
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class SlopeEntropyAlgorithm(OnlineAlgorithm):
    def __init__(
        self,
        window_size: int = 100,
        embedding_dim: int = 3,
        gamma: float = 1.0,
        delta: float = 1e-3,
        threshold: float = 0.3,
        normalize: bool = True,
    ):
        self._window_size = window_size
        self._embedding_dim = embedding_dim
        self._gamma = gamma
        self._delta = delta
        self._threshold = threshold
        self._normalize = normalize

        if delta >= gamma:
            raise ValueError(f"delta ({delta}) must be less than gamma ({gamma})")

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
        v = 2
        self._buffer.append(observation)
        self._position += 1

        min_required = self._window_size + self._embedding_dim - 1
        if len(self._buffer) < min_required:
            return

        current_window = np.array(list(self._buffer)[-self._window_size :])
        current_entropy = self._calculate_slope_entropy(current_window)

        if np.isinf(current_entropy) or np.isnan(current_entropy):
            current_entropy = 0.0

        self._entropy_values.append(current_entropy)

        if len(self._entropy_values) >= v:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])
            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

        if len(self._entropy_values) >= v * 5:
            recent_window = self._entropy_values[-5:]
            previous_window = self._entropy_values[-10:-5]
            recent_mean = np.mean(recent_window)
            previous_mean = np.mean(previous_window)

            if abs(recent_mean - previous_mean) > self._threshold * 0.8:
                self._last_change_point = self._position - 2

        if len(self._entropy_values) >= v * 4:
            recent_variance = np.var(self._entropy_values[-4:])
            previous_variance = np.var(self._entropy_values[-8:-4])
            if abs(recent_variance - previous_variance) > self._threshold * 0.5:
                self._last_change_point = self._position - 1

    def _calculate_slope_entropy(self, time_series: npt.NDArray[np.float64]) -> float:
        N = len(time_series)
        if self._embedding_dim > N:
            return 0.0

        pattern_counts: dict[tuple[int, ...], int] = {}
        total_patterns = 0

        for j in range(N - self._embedding_dim + 1):
            subsequence = time_series[j : j + self._embedding_dim]
            slope_pattern = self._create_slope_pattern(subsequence)
            pattern_key = tuple(slope_pattern)
            pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1
            total_patterns += 1

        if total_patterns == 0:
            return 0.0

        entropy = 0.0
        for count in pattern_counts.values():
            probability = count / total_patterns
            if probability > 0:
                entropy -= probability * np.log2(probability)

        if self._normalize:
            max_entropy = np.log2(len(pattern_counts)) if len(pattern_counts) > 1 else 1.0
            entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(entropy)

    def _create_slope_pattern(self, subsequence: npt.NDArray[np.float64]) -> list[int]:
        pattern = []
        for i in range(1, len(subsequence)):
            slope = subsequence[i] - subsequence[i - 1]

            if slope > self._gamma:
                symbol = 2
            elif slope > self._delta and slope <= self._gamma:
                symbol = 1
            elif abs(slope) <= self._delta:
                symbol = 0
            elif slope < -self._delta and slope >= -self._gamma:
                symbol = -1
            else:
                symbol = -2

            pattern.append(symbol)

        return pattern

    def get_entropy_history(self) -> list[float]:
        return self._entropy_values.copy()

    def get_current_parameters(self) -> dict[str, Any]:
        return {
            "window_size": self._window_size,
            "embedding_dim": self._embedding_dim,
            "gamma": self._gamma,
            "delta": self._delta,
            "threshold": self._threshold,
            "normalize": self._normalize,
            "max_symbols": 5,
            "max_patterns": 5 ** (self._embedding_dim - 1),
        }

    def set_parameters(
        self,
        embedding_dim: Optional[int] = None,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
        threshold: Optional[float] = None,
        normalize: Optional[bool] = None,
    ) -> None:
        if embedding_dim is not None:
            self._embedding_dim = embedding_dim
        if gamma is not None:
            self._gamma = gamma
        if delta is not None:
            self._delta = delta
        if threshold is not None:
            self._threshold = threshold
        if normalize is not None:
            self._normalize = normalize

        if self._delta >= self._gamma:
            raise ValueError(f"delta ({self._delta}) must be less than gamma ({self._gamma})")

    def get_pattern_distribution(self) -> dict[tuple[int, ...], float]:
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.array(list(self._buffer)[-self._window_size :])
        pattern_counts: dict[tuple[int, ...], int] = {}
        total_patterns = 0

        for j in range(len(current_window) - self._embedding_dim + 1):
            subsequence = current_window[j : j + self._embedding_dim]
            slope_pattern = self._create_slope_pattern(subsequence)
            pattern_key = tuple(slope_pattern)
            pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1
            total_patterns += 1

        pattern_probs = {}
        for pattern, count in pattern_counts.items():
            pattern_probs[pattern] = count / total_patterns if total_patterns > 0 else 0.0

        return pattern_probs

    def analyze_slope_characteristics(self) -> dict[str, Any]:
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.array(list(self._buffer)[-self._window_size :])
        slopes = np.diff(current_window)

        steep_positive = np.sum(slopes > self._gamma)
        gentle_positive = np.sum((slopes > self._delta) & (slopes <= self._gamma))
        flat = np.sum(np.abs(slopes) <= self._delta)
        gentle_negative = np.sum((slopes < -self._delta) & (slopes >= -self._gamma))
        steep_negative = np.sum(slopes < -self._gamma)
        total_slopes = len(slopes)

        return {
            "slope_entropy": self._calculate_slope_entropy(current_window),
            "steep_positive_ratio": steep_positive / total_slopes if total_slopes > 0 else 0,
            "gentle_positive_ratio": gentle_positive / total_slopes if total_slopes > 0 else 0,
            "flat_ratio": flat / total_slopes if total_slopes > 0 else 0,
            "gentle_negative_ratio": gentle_negative / total_slopes if total_slopes > 0 else 0,
            "steep_negative_ratio": steep_negative / total_slopes if total_slopes > 0 else 0,
            "slope_variance": np.var(slopes),
            "slope_mean": np.mean(slopes),
            "slope_std": np.std(slopes),
            "total_patterns": len(self.get_pattern_distribution()),
        }

    def get_symbol_meanings(self) -> dict[int, str]:
        return {
            2: f"Steep positive slope (> {self._gamma})",
            1: f"Gentle positive slope ({self._delta} to {self._gamma})",
            0: f"Flat/ties (≤ {self._delta})",
            -1: f"Gentle negative slope (-{self._gamma} to -{self._delta})",
            -2: f"Steep negative slope (< -{self._gamma})",
        }

    def demonstrate_encoding(
        self, sample_data: list[float]
    ) -> dict[str, Union[str, float, int, list[float], list[int], list[list[int]], dict[int, str]]]:
        if len(sample_data) < self._embedding_dim:
            return {"error": "Sample data too short"}

        sample_array = np.array(sample_data)
        slopes = np.diff(sample_array)

        symbols = []
        for slope in slopes:
            if slope > self._gamma:
                symbols.append(2)
            elif slope > self._delta and slope <= self._gamma:
                symbols.append(1)
            elif abs(slope) <= self._delta:
                symbols.append(0)
            elif slope < -self._delta and slope >= -self._gamma:
                symbols.append(-1)
            else:
                symbols.append(-2)

        patterns = []
        for j in range(len(sample_data) - self._embedding_dim + 1):
            subsequence = sample_array[j : j + self._embedding_dim]
            pattern = self._create_slope_pattern(subsequence)
            patterns.append(pattern)

        return {
            "original_data": sample_data,
            "slopes": slopes.tolist(),
            "symbols": symbols,
            "patterns": patterns,
            "slope_entropy": self._calculate_slope_entropy(sample_array),
            "encoding_rules": self.get_symbol_meanings(),
        }

    def reset(self) -> None:
        self._buffer.clear()
        self._entropy_values.clear()
        self._position = 0
        self._last_change_point = None
