from collections import deque
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import quad

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class TsallisEntropyAlgorithm(OnlineAlgorithm):
    def __init__(
        self,
        window_size: int = 100,
        q_parameter: float = 2.0,
        k_constant: float = 1.0,
        threshold: float = 0.1,
        num_bins: int = 20,
        use_kde: bool = False,
        normalize: bool = True,
        multi_q: bool = False,
    ):
        self._window_size = window_size
        self._q_parameter = q_parameter
        self._k_constant = k_constant
        self._threshold = threshold
        self._num_bins = num_bins
        self._use_kde = use_kde
        self._normalize = normalize
        self._multi_q = multi_q

        if window_size <= 0:
            raise ValueError("Window size must be positive")
        v = 1e-10
        if abs(q_parameter - 1.0) < v:
            raise ValueError("q parameter cannot be 1 (use Shannon entropy instead)")
        if k_constant <= 0:
            raise ValueError("k constant must be positive")
        if num_bins <= 1:
            raise ValueError("Number of bins must be greater than 1")

        if multi_q:
            self._q_values = [0.5, 1.5, 2.0, 2.5, 3.0]
        else:
            self._q_values = [q_parameter]

        self._buffer: deque[float] = deque(maxlen=window_size * 2)
        self._entropy_values: list[float] = []
        self._multi_entropy_values: dict[float, list[float]] = {q: [] for q in self._q_values}
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
        v1 = 2
        self._buffer.append(observation)
        self._position += 1

        if len(self._buffer) < self._window_size:
            return

        current_window = np.array(list(self._buffer)[-self._window_size :])

        if self._multi_q:
            entropies = {}
            for q in self._q_values:
                entropy = self._calculate_tsallis_entropy(current_window, q)
                entropies[q] = entropy
                self._multi_entropy_values[q].append(entropy)

            current_entropy = self._combine_multi_q_entropies(entropies)
        else:
            current_entropy = self._calculate_tsallis_entropy(current_window, self._q_parameter)

        if np.isinf(current_entropy) or np.isnan(current_entropy):
            current_entropy = 0.0

        self._entropy_values.append(current_entropy)

        if len(self._entropy_values) >= v1:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])

            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

        self._detect_entropy_trend()
        self._detect_entropy_variance()

        if self._multi_q:
            self._detect_multi_q_changes()

    def _calculate_tsallis_entropy(self, time_series: npt.NDArray[np.float64], q: float) -> float:
        if self._use_kde:
            return self._calculate_continuous_tsallis_entropy(time_series, q)
        else:
            return self._calculate_discrete_tsallis_entropy(time_series, q)

    def _calculate_discrete_tsallis_entropy(self, time_series: npt.NDArray[np.float64], q: float) -> float:
        hist, _ = np.histogram(time_series, bins=self._num_bins, density=False)
        probabilities = hist / np.sum(hist)
        probabilities = probabilities[probabilities > 0]

        if len(probabilities) == 0:
            return 0.0

        sum_p_q = np.sum(probabilities**q)

        v = 1e-10
        if abs(q - 1.0) < v:
            tsallis_entropy = -self._k_constant * np.sum(probabilities * np.log(probabilities))
        else:
            tsallis_entropy = self._k_constant * (1.0 / (q - 1.0)) * (1.0 - sum_p_q)

        if self._normalize:
            max_entropy = self._calculate_max_discrete_tsallis_entropy(len(probabilities), q)
            if max_entropy > 0:
                tsallis_entropy = tsallis_entropy / max_entropy

        return float(tsallis_entropy)

    def _calculate_continuous_tsallis_entropy(self, time_series: npt.NDArray[np.float64], q: float) -> float:
        try:
            kde = stats.gaussian_kde(time_series)
            data_min, data_max = np.min(time_series), np.max(time_series)
            margin = (data_max - data_min) * 0.2

            def integrand(x: Union[float, NDArray[np.float64]]) -> float:
                x_array: NDArray[np.float64] = np.atleast_1d(x).astype(np.float64)
                p_x = kde(x_array)[0]
                return float(p_x**q)

            integral_result, _ = quad(
                integrand, data_min - margin, data_max + margin, limit=100, epsabs=1e-8, epsrel=1e-8
            )
            v = 1e-10
            if abs(q - 1.0) < v:

                def shannon_integrand(x: Union[float, NDArray[np.float64]]) -> float:
                    x_array: NDArray[np.float64] = np.atleast_1d(x).astype(np.float64)
                    p_x = kde(x_array)[0]
                    return float(p_x * np.log(p_x + 1e-10))

                shannon_integral, _ = quad(
                    shannon_integrand, data_min - margin, data_max + margin, limit=100, epsabs=1e-8, epsrel=1e-8
                )
                tsallis_entropy = -self._k_constant * shannon_integral
            else:
                tsallis_entropy = (1.0 / (q - 1.0)) * (1.0 - integral_result)

            if np.isinf(tsallis_entropy) or np.isnan(tsallis_entropy):
                return 0.0

            return float(tsallis_entropy)

        except Exception:
            return self._calculate_discrete_tsallis_entropy(time_series, q)

    def _calculate_max_discrete_tsallis_entropy(self, n_states: int, q: float) -> float:
        v = 1e-10
        if n_states <= 1:
            return 0.0

        p_uniform = 1.0 / n_states

        if abs(q - 1.0) < v:
            return float(self._k_constant * np.log(n_states))
        else:
            return float(self._k_constant * (1.0 / (q - 1.0)) * (1.0 - n_states * (p_uniform**q)))

    def _combine_multi_q_entropies(self, entropies: dict[float, float]) -> float:
        weights = {}
        for q in entropies:
            if q < 1 or q > 1:
                weights[q] = 1.0
            else:
                weights[q] = 1.0

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        combined_entropy = sum(entropies[q] * weights[q] for q in entropies) / total_weight
        return combined_entropy

    def _detect_entropy_trend(self) -> None:
        v = 10
        if len(self._entropy_values) >= v:
            recent_entropies = self._entropy_values[-10:]
            x = np.arange(len(recent_entropies))
            slope, _, _, _, _ = stats.linregress(x, recent_entropies)

            if abs(slope) > self._threshold * 0.5:
                self._last_change_point = self._position - 5

    def _detect_entropy_variance(self) -> None:
        v = 20
        if len(self._entropy_values) >= v:
            first_half = self._entropy_values[-20:-10]
            second_half = self._entropy_values[-10:]

            var1 = np.var(first_half)
            var2 = np.var(second_half)

            if abs(var2 - var1) > self._threshold * 0.3:
                self._last_change_point = self._position - 10

    def _detect_multi_q_changes(self) -> None:
        if not self._multi_q:
            return

        change_votes = 0
        v = 2
        v1 = 5
        for q in self._q_values:
            if len(self._multi_entropy_values[q]) >= v1:
                recent_values = self._multi_entropy_values[q][-5:]
                if len(recent_values) >= v:
                    recent_diff = abs(recent_values[-1] - recent_values[-2])
                    if recent_diff > self._threshold * 0.7:
                        change_votes += 1

        if change_votes >= len(self._q_values) // 2 + 1:
            self._last_change_point = self._position - 2

    def get_entropy_history(self) -> list[float]:
        return self._entropy_values.copy()

    def get_multi_q_history(self) -> dict[float, list[float]]:
        return {q: values.copy() for q, values in self._multi_entropy_values.items()}

    def get_current_parameters(self) -> dict[str, Any]:
        return {
            "window_size": self._window_size,
            "q_parameter": self._q_parameter,
            "k_constant": self._k_constant,
            "threshold": self._threshold,
            "num_bins": self._num_bins,
            "use_kde": self._use_kde,
            "normalize": self._normalize,
            "multi_q": self._multi_q,
            "q_values": self._q_values.copy(),
        }

    def set_parameters(
        self,
        q_parameter: Optional[float] = None,
        k_constant: Optional[float] = None,
        threshold: Optional[float] = None,
        num_bins: Optional[int] = None,
        use_kde: Optional[bool] = None,
        normalize: Optional[bool] = None,
        multi_q: Optional[bool] = None,
    ) -> None:
        def set_q_param(q: float) -> None:
            v = 1e-10
            if abs(q - 1.0) < v:
                raise ValueError("q parameter cannot be 1")
            self._q_parameter = q
            if not self._multi_q:
                self._q_values = [q]

        def set_k_const(k: float) -> None:
            if k <= 0:
                raise ValueError("k constant must be positive")
            self._k_constant = k

        def set_num_bins_func(bins: int) -> None:
            if bins <= 1:
                raise ValueError("Number of bins must be greater than 1")
            self._num_bins = bins

        if q_parameter is not None:
            set_q_param(q_parameter)

        if k_constant is not None:
            set_k_const(k_constant)

        if threshold is not None:
            self._threshold = threshold

        if num_bins is not None:
            set_num_bins_func(num_bins)

        if use_kde is not None:
            self._use_kde = use_kde

        if normalize is not None:
            self._normalize = normalize

        if multi_q is not None:
            self._multi_q = multi_q
            if multi_q:
                self._q_values = [0.5, 1.5, 2.0, 2.5, 3.0]
                self._multi_entropy_values = {q: [] for q in self._q_values}
            else:
                self._q_values = [self._q_parameter]

    def analyze_q_sensitivity(self) -> dict[str, Any]:
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.array(list(self._buffer)[-self._window_size :])
        test_q_values = [0.1, 0.5, 1.5, 2.0, 2.5, 3.0, 5.0]
        q_entropies = {}

        for q in test_q_values:
            try:
                entropy = self._calculate_tsallis_entropy(current_window, q)
                q_entropies[q] = entropy
            except Exception:
                q_entropies[q] = 0.0

        entropy_ratios = {}
        base_q = 2.0
        if base_q in q_entropies:
            for q, entropy in q_entropies.items():
                if q != base_q and q_entropies[base_q] != 0:
                    entropy_ratios[q] = entropy / q_entropies[base_q]

        return {
            "q_entropies": q_entropies,
            "entropy_ratios": entropy_ratios,
            "most_sensitive_q": max(q_entropies.keys(), key=lambda x: abs(q_entropies[x])) if q_entropies else None,
            "entropy_variance_across_q": np.var(list(q_entropies.values())) if q_entropies else 0,
        }

    def get_complexity_metrics(self) -> dict[str, Any]:
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.array(list(self._buffer)[-self._window_size :])

        sub_extensive = self._calculate_tsallis_entropy(current_window, 0.5)
        extensive = self._calculate_discrete_tsallis_entropy(current_window, 1.01)
        super_extensive = self._calculate_tsallis_entropy(current_window, 2.5)

        return {
            "sub_extensive_entropy": sub_extensive,
            "extensive_entropy": extensive,
            "super_extensive_entropy": super_extensive,
            "entropy_ratio_sub_super": sub_extensive / super_extensive if super_extensive != 0 else float("inf"),
            "window_std": np.std(current_window),
            "window_mean": np.mean(current_window),
            "data_range": np.max(current_window) - np.min(current_window),
        }

    def reset(self) -> None:
        self._buffer.clear()
        self._entropy_values.clear()
        self._multi_entropy_values = {q: [] for q in self._q_values}
        self._position = 0
        self._last_change_point = None
