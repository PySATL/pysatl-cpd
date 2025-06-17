from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import stats

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class KLDivergenceAlgorithm(OnlineAlgorithm):
    def __init__(
        self,
        window_size: int = 100,
        reference_window_size: Optional[int] = None,
        threshold: float = 0.5,
        num_bins: int = 20,
        use_kde: bool = False,
        symmetric: bool = True,
        smoothing_factor: float = 1e-10,
    ):
        self._window_size = window_size
        self._reference_window_size = reference_window_size or window_size
        self._threshold = threshold
        self._num_bins = num_bins
        self._use_kde = use_kde
        self._symmetric = symmetric
        self._smoothing_factor = smoothing_factor

        if window_size <= 0 or self._reference_window_size <= 0:
            raise ValueError("Window sizes must be positive")
        if num_bins <= 1:
            raise ValueError("Number of bins must be greater than 1")
        if threshold <= 0:
            raise ValueError("Threshold must be positive")

        self._reference_buffer: deque[float] = deque(maxlen=self._reference_window_size)
        self._current_buffer: deque[float] = deque(maxlen=self._window_size)
        self._kl_values: list[float] = []
        self._position: int = 0
        self._last_change_point: Optional[int] = None
        self._reference_updated: bool = False

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
        v = 5
        self._current_buffer.append(observation)
        self._position += 1

        if len(self._reference_buffer) < self._reference_window_size:
            self._reference_buffer.append(observation)
            return

        if len(self._current_buffer) < self._window_size:
            return

        kl_divergence = self._calculate_kl_divergence()

        if np.isinf(kl_divergence) or np.isnan(kl_divergence):
            kl_divergence = 0.0

        self._kl_values.append(kl_divergence)

        if kl_divergence > self._threshold:
            self._last_change_point = self._position - self._window_size // 2
            self._update_reference_distribution()

        if len(self._kl_values) >= v:
            recent_kl = self._kl_values[-5:]
            kl_trend = np.mean(recent_kl)
            if kl_trend > self._threshold * 0.8:
                self._last_change_point = self._position - self._window_size // 4
                self._update_reference_distribution()

    def _calculate_kl_divergence(self) -> float:
        reference_data = np.array(list(self._reference_buffer))
        current_data = np.array(list(self._current_buffer))

        if self._use_kde:
            return self._calculate_kl_divergence_kde(reference_data, current_data)
        else:
            return self._calculate_kl_divergence_histogram(reference_data, current_data)

    def _calculate_kl_divergence_histogram(
        self, ref_data: npt.NDArray[np.float64], curr_data: npt.NDArray[np.float64]
    ) -> float:
        data_min = float(np.min(ref_data))
        data_max = float(np.max(ref_data))

        margin = (data_max - data_min) * 0.01
        bin_edges = np.linspace(data_min - margin, data_max + margin, self._num_bins + 1)

        ref_hist, _ = np.histogram(ref_data, bins=bin_edges, density=True)
        curr_hist, _ = np.histogram(curr_data, bins=bin_edges, density=True)

        ref_prob = ref_hist / np.sum(ref_hist) if np.sum(ref_hist) > 0 else ref_hist
        curr_prob = curr_hist / np.sum(curr_hist) if np.sum(curr_hist) > 0 else curr_hist

        ref_prob = ref_prob + self._smoothing_factor
        curr_prob = curr_prob + self._smoothing_factor

        ref_prob = ref_prob / np.sum(ref_prob)
        curr_prob = curr_prob / np.sum(curr_prob)

        kl_pq = float(np.sum(ref_prob * np.log(ref_prob / curr_prob)))

        if self._symmetric:
            kl_qp = float(np.sum(curr_prob * np.log(curr_prob / ref_prob)))
            return (kl_pq + kl_qp) / 2
        else:
            return kl_pq

    def _calculate_kl_divergence_kde(
        self, ref_data: npt.NDArray[np.float64], curr_data: npt.NDArray[np.float64]
    ) -> float:
        data_min = float(np.min(np.array([ref_data.min(), curr_data.min()])))
        data_max = float(np.max(np.array([ref_data.max(), curr_data.max()])))
        margin = (data_max - data_min) * 0.1
        x_eval = np.linspace(data_min - margin, data_max + margin, 1000)

        ref_kde = stats.gaussian_kde(ref_data)
        curr_kde = stats.gaussian_kde(curr_data)

        ref_density = ref_kde(x_eval)
        curr_density = curr_kde(x_eval)

        ref_density += self._smoothing_factor
        curr_density += self._smoothing_factor

        dx = x_eval[1] - x_eval[0]
        ref_density /= np.sum(ref_density) * dx
        curr_density /= np.sum(curr_density) * dx

        kl_pq = float(np.sum(ref_density * np.log(ref_density / curr_density)) * dx)

        if self._symmetric:
            kl_qp = float(np.sum(curr_density * np.log(curr_density / ref_density)) * dx)
            return (kl_pq + kl_qp) / 2
        else:
            return kl_pq

    def _update_reference_distribution(self) -> None:
        self._reference_buffer.clear()
        for value in self._current_buffer:
            self._reference_buffer.append(value)
        self._reference_updated = True

    def get_kl_history(self) -> list[float]:
        return self._kl_values.copy()

    def get_current_parameters(self) -> dict[str, float | int | bool]:
        return {
            "window_size": self._window_size,
            "reference_window_size": self._reference_window_size,
            "threshold": self._threshold,
            "num_bins": self._num_bins,
            "use_kde": self._use_kde,
            "symmetric": self._symmetric,
            "smoothing_factor": self._smoothing_factor,
        }

    def set_parameters(
        self,
        threshold: Optional[float] = None,
        num_bins: Optional[int] = None,
        use_kde: Optional[bool] = None,
        symmetric: Optional[bool] = None,
        smoothing_factor: Optional[float] = None,
    ) -> None:
        if threshold is not None:
            if threshold <= 0:
                raise ValueError("Threshold must be positive")
            self._threshold = threshold
        if num_bins is not None:
            if num_bins <= 1:
                raise ValueError("Number of bins must be greater than 1")
            self._num_bins = num_bins
        if use_kde is not None:
            self._use_kde = use_kde
        if symmetric is not None:
            self._symmetric = symmetric
        if smoothing_factor is not None:
            self._smoothing_factor = smoothing_factor

    def get_distribution_comparison(self) -> dict[str, float]:
        if len(self._reference_buffer) < self._reference_window_size or len(self._current_buffer) < self._window_size:
            return {}

        ref_data = np.array(list(self._reference_buffer))
        curr_data = np.array(list(self._current_buffer))

        ref_mean, ref_std = np.mean(ref_data), np.std(ref_data)
        curr_mean, curr_std = np.mean(curr_data), np.std(curr_data)

        kl_div = self._calculate_kl_divergence()
        ks_statistic, ks_pvalue = stats.ks_2samp(ref_data, curr_data)

        return {
            "kl_divergence": kl_div,
            "reference_mean": ref_mean,
            "reference_std": ref_std,
            "current_mean": curr_mean,
            "current_std": curr_std,
            "mean_difference": abs(curr_mean - ref_mean),
            "std_ratio": curr_std / ref_std if ref_std > 0 else float("inf"),
            "ks_statistic": ks_statistic,
            "ks_pvalue": ks_pvalue,
        }

    def analyze_distributions(self) -> dict[str, float]:
        if len(self._reference_buffer) < self._reference_window_size or len(self._current_buffer) < self._window_size:
            return {}

        ref_data = np.array(list(self._reference_buffer))
        curr_data = np.array(list(self._current_buffer))

        comparison = self.get_distribution_comparison()

        ref_entropy = stats.entropy(np.histogram(ref_data, bins=self._num_bins)[0] + self._smoothing_factor)
        curr_entropy = stats.entropy(np.histogram(curr_data, bins=self._num_bins)[0] + self._smoothing_factor)

        quantiles = [0.25, 0.5, 0.75]
        ref_quantiles = np.quantile(ref_data, quantiles)
        curr_quantiles = np.quantile(curr_data, quantiles)

        return {
            **comparison,
            "reference_entropy": ref_entropy,
            "current_entropy": curr_entropy,
            "entropy_difference": abs(curr_entropy - ref_entropy),
            "reference_quantiles": ref_quantiles.tolist(),
            "current_quantiles": curr_quantiles.tolist(),
            "quantile_differences": (np.abs(curr_quantiles - ref_quantiles)).tolist(),
        }

    def reset(self) -> None:
        self._reference_buffer.clear()
        self._current_buffer.clear()
        self._kl_values.clear()
        self._position = 0
        self._last_change_point = None
        self._reference_updated = False

    def force_reference_update(self) -> None:
        if len(self._current_buffer) >= self._window_size:
            self._update_reference_distribution()
