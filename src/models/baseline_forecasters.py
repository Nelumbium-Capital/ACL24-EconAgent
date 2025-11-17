"""
Baseline forecasting models for benchmarking.

Implements simple baseline models that serve as performance benchmarks:
- Naive: Last observed value repeated
- Trend: Linear trend extrapolation
"""
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from src.models.base_forecaster import BaseForecaster, ForecastResult
from src.utils.logging_config import logger


class NaiveForecaster(BaseForecaster):
    """
    Naive baseline forecaster.

    Forecasts future values as the last observed value (random walk model).
    This is a common benchmark in time series forecasting literature.

    Reference: Makridakis et al. (1982) "The accuracy of extrapolation methods"
    """

    def __init__(self, name: str = 'Naive'):
        """Initialize naive forecaster."""
        super().__init__(name)
        self.last_value = None
        self.historical_std = None

    def fit(self, data: pd.Series, **kwargs) -> 'NaiveForecaster':
        """
        Fit naive model (just stores last value and calculates historical volatility).

        Args:
            data: Historical time series as pandas Series with DatetimeIndex
            **kwargs: Additional parameters (unused)

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.name} model to {len(data)} observations")

        # Validate data
        data = self.validate_data(data)
        self.training_data = data

        # Store last value
        self.last_value = data.iloc[-1]

        # Calculate historical standard deviation for prediction intervals
        # Use first-difference standard deviation (common for random walk)
        if len(data) > 1:
            diffs = data.diff().dropna()
            self.historical_std = diffs.std()
        else:
            self.historical_std = 0.0

        self.is_fitted = True

        self.training_metadata = {
            'last_value': float(self.last_value),
            'historical_std': float(self.historical_std),
            'training_samples': len(data),
            'fit_date': datetime.now().isoformat()
        }

        logger.info(f"Naive model fitted. Last value: {self.last_value:.4f}, "
                   f"Historical std: {self.historical_std:.4f}")

        return self

    def forecast(
        self,
        horizon: int,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Generate naive forecasts (last value repeated).

        Args:
            horizon: Number of steps to forecast
            confidence_level: Confidence level for prediction intervals

        Returns:
            ForecastResult with point forecasts and intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        logger.info(f"Generating {horizon}-step naive forecast")

        # Point forecast: repeat last value
        point_forecast = np.full(horizon, self.last_value)

        # Prediction intervals widen with forecast horizon (random walk property)
        # σ_h = σ_1 * sqrt(h) for random walk
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        lower_bound = np.zeros(horizon)
        upper_bound = np.zeros(horizon)

        for h in range(horizon):
            std_h = self.historical_std * np.sqrt(h + 1)
            lower_bound[h] = self.last_value - z_score * std_h
            upper_bound[h] = self.last_value + z_score * std_h

        result = ForecastResult(
            model_name=self.name,
            series_name=self.training_data.name if self.training_data.name else 'series',
            forecast_date=datetime.now(),
            horizon=horizon,
            point_forecast=point_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            metadata={
                'last_value': float(self.last_value),
                'model_type': 'naive_baseline'
            }
        )

        logger.info(f"Naive forecast generated successfully")

        return result


class TrendForecaster(BaseForecaster):
    """
    Linear trend baseline forecaster.

    Extrapolates a linear trend fitted to historical data.
    Uses ordinary least squares regression on time index.

    Reference: Hyndman & Athanasopoulos (2021) "Forecasting: Principles and Practice"
    """

    def __init__(self, name: str = 'Trend'):
        """Initialize trend forecaster."""
        super().__init__(name)
        self.slope = None
        self.intercept = None
        self.trend_std = None

    def fit(self, data: pd.Series, **kwargs) -> 'TrendForecaster':
        """
        Fit linear trend model using OLS regression.

        Args:
            data: Historical time series as pandas Series with DatetimeIndex
            **kwargs: Additional parameters (unused)

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.name} model to {len(data)} observations")

        # Validate data
        data = self.validate_data(data)
        self.training_data = data

        # Create time index (0, 1, 2, ...)
        t = np.arange(len(data))
        y = data.values

        # Fit linear trend: y = slope * t + intercept
        self.slope, self.intercept, r_value, p_value, std_err = stats.linregress(t, y)

        # Calculate residual standard deviation for prediction intervals
        fitted_values = self.intercept + self.slope * t
        residuals = y - fitted_values
        self.trend_std = np.std(residuals)

        self.is_fitted = True

        self.training_metadata = {
            'slope': float(self.slope),
            'intercept': float(self.intercept),
            'r_squared': float(r_value ** 2),
            'residual_std': float(self.trend_std),
            'training_samples': len(data),
            'fit_date': datetime.now().isoformat()
        }

        logger.info(f"Trend model fitted. Slope: {self.slope:.6f}, "
                   f"Intercept: {self.intercept:.4f}, R²: {r_value**2:.4f}")

        return self

    def forecast(
        self,
        horizon: int,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Generate trend forecasts by extrapolating fitted line.

        Args:
            horizon: Number of steps to forecast
            confidence_level: Confidence level for prediction intervals

        Returns:
            ForecastResult with point forecasts and intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        logger.info(f"Generating {horizon}-step trend forecast")

        # Forecast time indices
        n = len(self.training_data)
        future_t = np.arange(n, n + horizon)

        # Point forecast: extrapolate trend
        point_forecast = self.intercept + self.slope * future_t

        # Prediction intervals (standard error of forecast)
        # SE increases with distance from sample mean
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        # Simple approach: constant standard error equal to residual std
        # More sophisticated: account for extrapolation uncertainty
        lower_bound = point_forecast - z_score * self.trend_std
        upper_bound = point_forecast + z_score * self.trend_std

        result = ForecastResult(
            model_name=self.name,
            series_name=self.training_data.name if self.training_data.name else 'series',
            forecast_date=datetime.now(),
            horizon=horizon,
            point_forecast=point_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            metadata={
                'slope': float(self.slope),
                'intercept': float(self.intercept),
                'model_type': 'linear_trend'
            }
        )

        logger.info(f"Trend forecast generated successfully")

        return result
