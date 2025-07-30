from collections import deque
from typing import Any, Optional

import numpy as np
import numpy.typing as npt


class SDARmodel:
    """
    Implements a Seasonal Vector Autoregressive (SDAR) model for online calculation
    of anomaly scores in time series.
    """

    def __init__(self, order: int = 5, forgetting_factor: float = 0.99, smoothing_window_size: int = 5) -> None:
        """
        Initializes the SDAR model.

        :param order: The order of the autoregressive model (p). Must be a positive integer.
        :param forgetting_factor: The forgetting factor (lambda) in the range (0, 1).
                                Values close to 1 imply slow forgetting.
        :param smoothing_window_size: The size of the window for averaging (smoothing) the anomaly scores.
                                Must be a positive integer.
        :return:
        """
        assert order > 0, "Order must be a positive integer."
        assert 0 < forgetting_factor < 1, "Forgetting factor must be between 0 and 1."
        assert smoothing_window_size > 0, "Smoothing window size must be a positive integer."

        self.__order = order
        self.__lambda = forgetting_factor
        self.__smoothing_window_size: int = smoothing_window_size

        self.__dimension: Optional[int] = None
        self.__mu: Optional[npt.NDArray[np.float64]] = None
        self.__covariance_matrices: deque[npt.NDArray[np.float64]] = deque(maxlen=self.__order + 1)
        self.__ar_matrices: list[npt.NDArray[np.float64]] = []
        self.__sigma: Optional[npt.NDArray[np.float64]] = None

        self.__history: deque[npt.NDArray[np.float64]] = deque(maxlen=self.__order)
        self.__scores_buffer: deque[np.float64] = deque(maxlen=self.__smoothing_window_size)

    def __initialize_state(self, x: npt.NDArray[np.float64]) -> None:
        """
        Initializes the model's internal state based on the first observation.
        :param x: the first observation.
        :return:
        """
        self.__dimension = x.shape[0]
        self.__mu = np.zeros(self.__dimension)

        for _ in range(self.__order + 1):
            self.__covariance_matrices.append(np.zeros((self.__dimension, self.__dimension)))

        self.__ar_matrices = [np.zeros((self.__dimension, self.__dimension)) for _ in range(self.__order)]
        self.__sigma = np.eye(self.__dimension)

    def clear(self) -> None:
        """
        Resets the internal state of the model to its initial values.
        :return:
        """
        if self.__dimension is None:
            return

        assert self.__mu is not None, "clear() should not be called on an uninitialized model with mu=None"

        self.__mu.fill(0)
        for i in range(len(self.__covariance_matrices)):
            self.__covariance_matrices[i].fill(0)
        self.__ar_matrices = [np.zeros((self.__dimension, self.__dimension)) for _ in range(self.__order)]
        self.__sigma = np.eye(self.__dimension)
        self.__history.clear()
        self.__scores_buffer.clear()

    def __calculate_error(self, x: npt.NDArray[np.float64]) -> Any:
        """
        Calculates the prediction error for the current observation.
        :param x: a new observation.
        :return: the error vector between the observation and the predicted point.
        """
        assert self.__mu is not None, "Model must be initialized before calculating error."

        centered_history = np.array(list(self.__history)) - self.__mu
        prediction_offset = np.sum([self.__ar_matrices[i] @ centered_history[i] for i in range(self.__order)], axis=0)
        x_hat = self.__mu + prediction_offset
        error = x - x_hat
        return error

    def __update__matrices(self, x: npt.NDArray[np.float64]) -> None:
        """
        Recursively updates the covariance matrices using the forgetting factor.
        :param x: a new observation.
        :return:
        """
        assert self.__mu is not None, "Model must be initialized before update matrices."

        centered_x = x - self.__mu
        self.__covariance_matrices[0] = self.__lambda * self.__covariance_matrices[0] + (1 - self.__lambda) * np.outer(
            centered_x, centered_x
        )
        for i in range(1, self.__order + 1):
            centered_hist = self.__history[-i] - self.__mu
            self.__covariance_matrices[i] = self.__lambda * self.__covariance_matrices[i] + (
                1 - self.__lambda
            ) * np.outer(centered_x, centered_hist)

    def __update_model(self, x: npt.NDArray[np.float64]) -> None:
        """
        Performs a single model update step based on a new observation.
        :param x: a new observation.
        :return:
        """
        if self.__dimension is None:
            self.__initialize_state(x)

        assert self.__mu is not None
        if len(self.__history) < self.__order:
            self.__mu = self.__lambda * self.__mu + (1 - self.__lambda) * x
            self.__history.append(x)
            return

        self.__mu = self.__lambda * self.__mu + (1 - self.__lambda) * x

        self.__update__matrices(x)
        self.__solve_equations()

        error = self.__calculate_error(x)

        assert self.__sigma is not None
        self.__sigma = self.__lambda * self.__sigma + (1 - self.__lambda) * np.outer(error, error)

        score = self.__calculate_log_likelihood(error)
        score = np.maximum(score, 1e-6)
        self.__scores_buffer.append(score)
        self.__history.append(x)
        return

    def __calculate_log_likelihood(self, error: npt.NDArray[np.float64]) -> np.float64:
        """
        Calculates the anomaly score as the negative log-likelihood of the prediction error.
        :param error: the prediction error.
        :return: the calculated anomaly score.
        """
        assert self.__dimension is not None and self.__sigma is not None, "Model must be initialized."
        _, log_det = np.linalg.slogdet(self.__sigma)

        inv_sigma = np.linalg.inv(self.__sigma)
        mahalanobis_dist = error.T @ inv_sigma @ error

        log_likelihood = -0.5 * (self.__dimension * np.log(2 * np.pi) + log_det + mahalanobis_dist)
        return np.float64(-log_likelihood)

    def __solve_equations(self) -> None:
        """
        Finds the AR model coefficients by solving the equations for the multivariate case.
        Handles potential singularity of the matrix.
        :return:
        """
        assert self.__dimension is not None, "Model must be initialized before solve equations."

        d, p = self.__dimension, self.__order
        transition_matrix = np.zeros((p * d, p * d))
        for i in range(p):
            for j in range(p):
                lag = abs(i - j)
                cov_mat = self.__covariance_matrices[lag]
                transition_matrix[i * d : (i + 1) * d, j * d : (j + 1) * d] = cov_mat if i >= j else cov_mat.T

        transition_matrix_stacked = np.vstack([self.__covariance_matrices[i + 1] for i in range(p)])

        weights: npt.NDArray[np.float64] = np.linalg.solve(transition_matrix, transition_matrix_stacked).astype(
            np.float64
        )
        self.__ar_matrices = [weights[i * d : (i + 1) * d, :] for i in range(p)]

    def get_smoothed_score(self, x: npt.NDArray[np.float64]) -> Optional[np.float64]:
        """
        Processes a new observation and returns the smoothed anomaly score.
        :param x: new observation.
        :return: the smoothed anomaly score, or None if the smoothing buffer is not yet full.
        """
        self.__update_model(x)
        if len(self.__scores_buffer) < self.__smoothing_window_size:
            return None

        score = np.mean(list(self.__scores_buffer))
        score = np.maximum(score, 1e-9)
        return np.float64(score)
