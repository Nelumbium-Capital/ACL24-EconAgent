"""
ARIMA and SARIMA forecasting models using statsmodels.
"""
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

from src.models.base_forecaster import BaseForecaster, ForecastResult
from src.utils.logging_config import logger


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA (AutoRegressive Integrated Moving Average) forecaster.
    
    Supports automatic order selection using AIC/BIC criteria.
    """
    
    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        auto_order: bool = True,
        seasonal: bool = False,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        information_criterion: str = 'aic',
        name: str = 'ARIMA'
    ):
        """
        Initialize ARIMA forecaster.
        
        Args:
            order: ARIMA order (p, d, q). If None and auto_order=True, will be determined automatically
            auto_order: Whether to automatically determine optimal order
            seasonal: Whether to use SARIMA (seasonal ARIMA)
            seasonal_order: Seasonal order (P, D, Q, s) for SARIMA
            information_criterion: Criterion for model selection ('aic' or 'bic')
            name: Name identifier for this forecaster
        """
        super().__init__(name)
        self.order = order
        self.auto_order = auto_order
        self.seasonal = seasonal
        self.seasonal_order = seasonal_order
        self.information_criterion = information_criterion
        self.model = None
        self.fitted_model = None
        
    def fit(self, data: pd.Series, **kwargs) -> 'ARIMAForecaster':
        """
        Fit ARIMA model to historical data.
        
        Args:
            data: Historical time series data
            **kwargs: Additional parameters for model fitting
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.name} model to {len(data)} observations")
        
        # Validate data
        data = self.validate_data(data)
        self.training_data = data
        
        # Determine order if auto_order is enabled
        if self.auto_order and self.order is None:
            logger.info("Automatically determining optimal ARIMA order")
            self.order = self._auto_select_order(data)
            logger.info(f"Selected order: {self.order}")
        
        # Default order if not specified
        if self.order is None:
            self.order = (1, 1, 1)
            logger.warning(f"No order specified, using default: {self.order}")
        
        try:
            # Fit model
            if self.seasonal and self.seasonal_order is not None:
                logger.info(f"Fitting SARIMA model with seasonal order: {self.seasonal_order}")
                self.model = SARIMAX(
                    data,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                logger.info(f"Fitting ARIMA model with order: {self.order}")
                self.model = ARIMA(
                    data,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            # Store training metadata
            self.training_metadata = {
                'order': self.order,
                'seasonal_order': self.seasonal_order if self.seasonal else None,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'training_samples': len(data),
                'fit_date': datetime.now().isoformat()
            }
            
            logger.info(f"Model fitted successfully. AIC: {self.fitted_model.aic:.2f}, BIC: {self.fitted_model.bic:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to fit ARIMA model: {e}")
            raise
        
        return self
    
    def forecast(
        self,
        horizon: int,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Generate forecasts for future periods.
        
        Args:
            horizon: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            ForecastResult with point forecasts and intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        logger.info(f"Generating {horizon}-step forecast with {confidence_level*100}% confidence")
        
        try:
            # Generate forecast
            forecast_result = self.fitted_model.get_forecast(steps=horizon)
            
            # Extract point forecasts
            point_forecast = forecast_result.predicted_mean.values
            
            # Get prediction intervals
            pred_intervals = forecast_result.conf_int(alpha=1-confidence_level)
            lower_bound = pred_intervals.iloc[:, 0].values
            upper_bound = pred_intervals.iloc[:, 1].values
            
            # Create result object
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
                    'order': self.order,
                    'seasonal_order': self.seasonal_order if self.seasonal else None,
                    'aic': self.fitted_model.aic,
                    'bic': self.fitted_model.bic
                }
            )
            
            logger.info(f"Forecast generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate forecast: {e}")
            raise
    
    def _auto_select_order(
        self,
        data: pd.Series,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        seasonal: bool = False,
        m: int = 12
    ) -> Tuple[int, int, int]:
        """
        Automatically select optimal ARIMA order using pmdarima.
        
        Args:
            data: Time series data
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            seasonal: Whether to consider seasonal components
            m: Seasonal period
            
        Returns:
            Optimal order (p, d, q)
        """
        if not HAS_PMDARIMA:
            logger.warning("pmdarima not available, using default order (1,1,1)")
            return (1, 1, 1)
        
        try:
            # Use auto_arima for order selection
            auto_model = pm.auto_arima(
                data,
                start_p=0, max_p=max_p,
                start_q=0, max_q=max_q,
                max_d=max_d,
                seasonal=seasonal,
                m=m if seasonal else 1,
                information_criterion=self.information_criterion,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            order = auto_model.order
            
            # Store seasonal order if seasonal
            if seasonal:
                self.seasonal_order = auto_model.seasonal_order
                self.seasonal = True
            
            return order
            
        except Exception as e:
            logger.warning(f"Auto order selection failed: {e}, using default (1,1,1)")
            return (1, 1, 1)
    
    def get_residuals(self) -> np.ndarray:
        """
        Get model residuals from fitted model.
        
        Returns:
            Array of residuals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.fitted_model.resid
    
    def plot_diagnostics(self):
        """
        Plot diagnostic plots for the fitted model.
        
        Returns:
            Matplotlib figure with diagnostic plots
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.fitted_model.plot_diagnostics(figsize=(12, 8))


class SARIMAForecaster(ARIMAForecaster):
    """
    SARIMA (Seasonal ARIMA) forecaster.
    
    Extends ARIMA to handle seasonal patterns in time series data.
    """
    
    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        auto_order: bool = True,
        seasonal_period: int = 12,
        information_criterion: str = 'aic',
        name: str = 'SARIMA'
    ):
        """
        Initialize SARIMA forecaster.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            auto_order: Whether to automatically determine optimal orders
            seasonal_period: Number of periods in a season (e.g., 12 for monthly data)
            information_criterion: Criterion for model selection ('aic' or 'bic')
            name: Name identifier for this forecaster
        """
        super().__init__(
            order=order,
            auto_order=auto_order,
            seasonal=True,
            seasonal_order=seasonal_order,
            information_criterion=information_criterion,
            name=name
        )
        self.seasonal_period = seasonal_period
        
        # Set default seasonal order if not provided
        if self.seasonal_order is None and not auto_order:
            self.seasonal_order = (1, 1, 1, seasonal_period)
    
    def fit(self, data: pd.Series, **kwargs) -> 'SARIMAForecaster':
        """
        Fit SARIMA model to historical data.
        
        Args:
            data: Historical time series data
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting SARIMA model with seasonal period: {self.seasonal_period}")
        
        # Validate data
        data = self.validate_data(data)
        
        # Check if we have enough data for seasonal modeling
        if len(data) < 2 * self.seasonal_period:
            logger.warning(
                f"Data length ({len(data)}) is less than 2 seasonal periods. "
                f"Consider using non-seasonal ARIMA instead."
            )
        
        # Perform seasonal decomposition for diagnostics
        self._analyze_seasonality(data)
        
        # Call parent fit method
        return super().fit(data, **kwargs)
    
    def _analyze_seasonality(self, data: pd.Series) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in the data.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with seasonality analysis results
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        try:
            # Perform seasonal decomposition
            if len(data) >= 2 * self.seasonal_period:
                decomposition = seasonal_decompose(
                    data,
                    model='additive',
                    period=self.seasonal_period,
                    extrapolate_trend='freq'
                )
                
                # Calculate strength of seasonality
                seasonal_strength = 1 - (
                    np.var(decomposition.resid.dropna()) /
                    np.var(decomposition.seasonal.dropna() + decomposition.resid.dropna())
                )
                
                logger.info(f"Seasonal strength: {seasonal_strength:.3f}")
                
                self.training_metadata['seasonal_strength'] = seasonal_strength
                self.training_metadata['has_trend'] = np.abs(decomposition.trend.dropna()).mean() > 0
                
                return {
                    'seasonal_strength': seasonal_strength,
                    'decomposition': decomposition
                }
            else:
                logger.warning("Insufficient data for seasonal decomposition")
                return {}
                
        except Exception as e:
            logger.warning(f"Seasonal decomposition failed: {e}")
            return {}
    
    def _auto_select_order(
        self,
        data: pd.Series,
        max_p: int = 3,
        max_d: int = 2,
        max_q: int = 3,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2
    ) -> Tuple[int, int, int]:
        """
        Automatically select optimal SARIMA orders.
        
        Args:
            data: Time series data
            max_p: Maximum non-seasonal AR order
            max_d: Maximum non-seasonal differencing order
            max_q: Maximum non-seasonal MA order
            max_P: Maximum seasonal AR order
            max_D: Maximum seasonal differencing order
            max_Q: Maximum seasonal MA order
            
        Returns:
            Optimal non-seasonal order (p, d, q)
        """
        if not HAS_PMDARIMA:
            logger.warning("pmdarima not available, using default orders")
            self.seasonal_order = (1, 1, 1, self.seasonal_period)
            return (1, 1, 1)
        
        try:
            logger.info("Automatically selecting SARIMA orders")
            
            # Use auto_arima with seasonal components
            auto_model = pm.auto_arima(
                data,
                start_p=0, max_p=max_p,
                start_q=0, max_q=max_q,
                max_d=max_d,
                start_P=0, max_P=max_P,
                start_Q=0, max_Q=max_Q,
                max_D=max_D,
                seasonal=True,
                m=self.seasonal_period,
                information_criterion=self.information_criterion,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            order = auto_model.order
            self.seasonal_order = auto_model.seasonal_order
            
            logger.info(f"Selected orders - ARIMA: {order}, Seasonal: {self.seasonal_order}")
            
            return order
            
        except Exception as e:
            logger.warning(f"Auto order selection failed: {e}, using defaults")
            self.seasonal_order = (1, 1, 1, self.seasonal_period)
            return (1, 1, 1)
