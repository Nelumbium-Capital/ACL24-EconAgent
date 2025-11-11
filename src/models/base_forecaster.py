"""
Base forecaster interface and common utilities for time-series forecasting.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging_config import logger
from src.utils.error_handler import error_handler, handle_errors, ErrorCategory, ErrorSeverity, ModelError


@dataclass
class ForecastResult:
    """Result from a forecasting model."""
    model_name: str
    series_name: str
    forecast_date: datetime
    horizon: int
    point_forecast: np.ndarray
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    confidence_level: float = 0.95
    metadata: Optional[Dict[str, Any]] = None


class BaseForecaster(ABC):
    """
    Abstract base class for time-series forecasting models.
    
    All forecasting models should inherit from this class and implement
    the fit() and forecast() methods.
    """
    
    def __init__(self, name: str):
        """
        Initialize base forecaster.
        
        Args:
            name: Name identifier for this forecaster
        """
        self.name = name
        self.is_fitted = False
        self.training_data = None
        self.training_metadata = {}
        
    @abstractmethod
    def fit(self, data: pd.Series, **kwargs) -> 'BaseForecaster':
        """
        Fit the forecasting model to historical data.
        
        Args:
            data: Historical time series data as pandas Series with DatetimeIndex
            **kwargs: Additional model-specific parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def forecast(
        self, 
        horizon: int,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Generate forecasts for future time periods.
        
        Args:
            horizon: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals (0-1)
            
        Returns:
            ForecastResult containing point forecasts and prediction intervals
        """
        pass
    
    def validate_data(self, data: pd.Series) -> pd.Series:
        """
        Validate and prepare input data for forecasting.
        
        Args:
            data: Input time series data
            
        Returns:
            Validated and cleaned data
            
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Data must be a pandas Series")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")
        
        if len(data) < 2:
            raise ValueError("Data must contain at least 2 observations")
        
        # Check for missing values
        if data.isna().any():
            logger.warning(f"Data contains {data.isna().sum()} missing values, forward-filling")
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Check for infinite values
        if np.isinf(data.values).any():
            raise ValueError("Data contains infinite values")
        
        # Sort by index
        data = data.sort_index()
        
        return data
    
    def check_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with test results
        """
        from statsmodels.tsa.stattools import adfuller
        
        result = adfuller(data.dropna(), autolag='AIC')
        
        is_stationary = result[1] < 0.05  # p-value < 0.05
        
        return {
            'is_stationary': is_stationary,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4]
        }
    
    def difference_series(
        self, 
        data: pd.Series, 
        order: int = 1,
        seasonal_period: Optional[int] = None
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Difference time series to achieve stationarity.
        
        Args:
            data: Time series data
            order: Order of differencing
            seasonal_period: Period for seasonal differencing
            
        Returns:
            Tuple of (differenced series, transformation info)
        """
        differenced = data.copy()
        transform_info = {
            'order': order,
            'seasonal_period': seasonal_period,
            'original_values': data.iloc[:order].values if order > 0 else None
        }
        
        # Regular differencing
        for _ in range(order):
            differenced = differenced.diff().dropna()
        
        # Seasonal differencing
        if seasonal_period is not None:
            transform_info['seasonal_values'] = differenced.iloc[:seasonal_period].values
            differenced = differenced.diff(seasonal_period).dropna()
        
        return differenced, transform_info
    
    def inverse_difference(
        self,
        forecasts: np.ndarray,
        transform_info: Dict[str, Any],
        last_values: pd.Series
    ) -> np.ndarray:
        """
        Inverse differencing transformation to get forecasts in original scale.
        
        Args:
            forecasts: Differenced forecasts
            transform_info: Information about differencing transformation
            last_values: Last values from original series
            
        Returns:
            Forecasts in original scale
        """
        result = forecasts.copy()
        
        # Inverse seasonal differencing
        if transform_info.get('seasonal_period') is not None:
            seasonal_period = transform_info['seasonal_period']
            seasonal_base = last_values.iloc[-seasonal_period:].values
            
            for i in range(len(result)):
                if i < seasonal_period:
                    result[i] = result[i] + seasonal_base[i]
                else:
                    result[i] = result[i] + result[i - seasonal_period]
        
        # Inverse regular differencing
        order = transform_info.get('order', 0)
        if order > 0:
            last_value = last_values.iloc[-1]
            for i in range(len(result)):
                if i == 0:
                    result[i] = result[i] + last_value
                else:
                    result[i] = result[i] + result[i - 1]
        
        return result
    
    def calculate_prediction_intervals(
        self,
        point_forecast: np.ndarray,
        residuals: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals using residual standard error.
        
        Args:
            point_forecast: Point forecasts
            residuals: Model residuals from training
            confidence_level: Confidence level (0-1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Calculate standard error
        std_error = np.std(residuals)
        
        # Get z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Calculate intervals (expanding with forecast horizon)
        horizon = len(point_forecast)
        margin = np.array([std_error * z_score * np.sqrt(i + 1) for i in range(horizon)])
        
        lower_bound = point_forecast - margin
        upper_bound = point_forecast + margin
        
        return lower_bound, upper_bound
    
    def evaluate_forecast(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate forecast accuracy using multiple metrics.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        # Calculate metrics
        errors = actual - predicted
        
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        
        # MAPE (avoid division by zero)
        non_zero_mask = actual != 0
        if non_zero_mask.any():
            mape = np.mean(np.abs(errors[non_zero_mask] / actual[non_zero_mask])) * 100
        else:
            mape = np.nan
        
        # Directional accuracy
        if len(actual) > 1:
            actual_direction = np.sign(np.diff(actual))
            predicted_direction = np.sign(np.diff(predicted))
            directional_accuracy = np.mean(actual_direction == predicted_direction)
        else:
            directional_accuracy = np.nan
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'training_samples': len(self.training_data) if self.training_data is not None else 0,
            'metadata': self.training_metadata
        }
    
    def __repr__(self) -> str:
        """String representation of the forecaster."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {fitted_status})"
