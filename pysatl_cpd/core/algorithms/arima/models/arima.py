import warnings
from typing import Any, Optional

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


class ArimaModel:
    """
    A wrapper for the statsmodels ARIMA model to simplify its use in online
    change point detection tasks.
    """

    def __init__(self) -> None:
        """
        Initializes the ArimaModel instance, without any concrete values.
        """
        self.__training_data: list[np.float64] = []
        self.__results: Optional[ARIMAResults] = None

    def clear(self) -> None:
        """
        Resets the model's state by clearing training data and results.
        :return:
        """
        self.__training_data = []
        self.__results = None

    def fit(self, training_data: list[np.float64]) -> Any:
        """
        Fits the ARIMA model on the provided training data.

        If the data variance is very low, it fits the model as a simple mean model.
        Otherwise, the standard ARIMA(0,0,0) model is used, which is suitable for stationary data.
        Convergence warnings are suppressed to avoid cluttering output.

        :param training_data: the time series data to train the model on.
        :return: the residuals of the model after fitting.
        """
        self.__training_data = training_data

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = ARIMA(self.__training_data, order=(0, 0, 0))
            self.__results = model.fit()

        return self.__results.resid

    def predict(self, steps: int) -> Any:
        """
        Forecasts future values of the time series.
        :param steps: the number of steps to forecast ahead.
        :return: a list of forecasted values.
        """
        assert self.__results is not None, "Model must be fitted before prediction."

        return self.__results.forecast(steps=steps)

    def update(self, observation: list[np.float64]) -> Any:
        """
        Appends a new observation to the fitted ARIMA model.
        :param observation: the new observation to add to the model.
        :return:
        """
        assert self.__results is not None, "Model must be fitted before prediction."

        self.__results = self.__results.append(observation)
