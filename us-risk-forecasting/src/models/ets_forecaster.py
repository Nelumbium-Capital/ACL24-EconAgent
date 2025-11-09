"""
Exponential Smoothing (ETS) forecasting models using statsmodels.
"""
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from src.models.base_forecaster import BaseForecaster, ForecastResult
from src.utils.logging_config import logger


class ETSForecaster(BaseForecaster):
    """
    ETS (Error, Trend, Seasonal) Exponential Smoothing forecaster.
    
    Supports various combinations of error, trend, and seasonal components.
    """
    
    def __init__(
        self,
        trend: Optional[str] = 'add',
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        damped_trend: bool = False,
        use_boxcox: bool = False,
        initialization_method: str = 'estimated',
        name: str = 'ETS'
    ):
        """
        Initialize ETS forecaster.
        
        Args:
            trend: Type of trend component ('add', 'mul', or None)
            seasonal: Type of seasonal component ('add', 'mul', or None)
            seasonal_periods: Number of periods in a complete seasonal cycle
            damped_trend: Whether to use damped trend
            use_boxcox: Whether to apply Box-Cox transformation
            initialization_method: Method for initialization ('estimated', 'heuristic', 'known')
            name: Name identifier for this forecaster
        """
        super().__init__(name)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.use_boxcox = use_boxcox
        self.initialization_method = initialization_method
        self.model = None
        self.fitted_model = None
        
    def fit(self, data: pd.Series, **kwargs) -> 'ETSForecaster':
        """
        Fit ETS model to historical data.
        
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
        
        # Check for negative values if using multiplicative components
        if (self.trend == 'mul' or self.seasonal == 'mul') and (data <= 0).any():
            logger.warning("Data contains non-positive values, switching to additive components")
            if self.trend == 'mul':
                self.trend = 'add'
            if self.seasonal == 'mul':
                self.seasonal = 'add'
        
        # Validate seasonal periods
        if self.seasonal is not None and self.seasonal_periods is None:
            logger.warning("Seasonal component specified but no seasonal_periods given, inferring from data")
            self.seasonal_periods = self._infer_seasonal_periods(data)
        
        # Check if we have enough data for seasonal modeling
        if self.seasonal is not None and len(data) < 2 * self.seasonal_periods:
            logger.warning(
                f"Insufficient data for seasonal modeling (need at least {2 * self.seasonal_periods} points). "
                f"Disabling seasonal component."
            )
            self.seasonal = None
        
        try:
            # Create and fit model
            logger.info(
                f"Fitting ETS model: trend={self.trend}, seasonal={self.seasonal}, "
                f"seasonal_periods={self.seasonal_periods}, damped={self.damped_trend}"
            )
            
            self.model = ExponentialSmoothing(
                data,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend,
                use_boxcox=self.use_boxcox,
                initialization_method=self.initialization_method
            )
            
            self.fitted_model = self.model.fit(optimized=True)
            self.is_fitted = True
            
            # Store training metadata
            self.training_metadata = {
                'trend': self.trend,
                'seasonal': self.seasonal,
                'seasonal_periods': self.seasonal_periods,
                'damped_trend': self.damped_trend,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'training_samples': len(data),
                'fit_date': datetime.now().isoformat(),
                'smoothing_level': self.fitted_model.params.get('smoothing_level'),
                'smoothing_trend': self.fitted_model.params.get('smoothing_trend'),
                'smoothing_seasonal': self.fitted_model.params.get('smoothing_seasonal')
            }
            
            logger.info(
                f"Model fitted successfully. AIC: {self.fitted_model.aic:.2f}, "
                f"BIC: {self.fitted_model.bic:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to fit ETS model: {e}")
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
            forecast_result = self.fitted_model.forecast(steps=horizon)
            point_forecast = forecast_result.values
            
            # Calculate prediction intervals using simulation
            simulations = self.fitted_model.simulate(
                nsimulations=horizon,
                repetitions=1000,
                random_errors='bootstrap'
            )
            
            # Calculate percentiles for confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(simulations, lower_percentile, axis=1)
            upper_bound = np.percentile(simulations, upper_percentile, axis=1)
            
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
                    'trend': self.trend,
                    'seasonal': self.seasonal,
                    'seasonal_periods': self.seasonal_periods,
                    'damped_trend': self.damped_trend,
                    'aic': self.fitted_model.aic,
                    'bic': self.fitted_model.bic
                }
            )
            
            logger.info("Forecast generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate forecast: {e}")
            raise
    
    def _infer_seasonal_periods(self, data: pd.Series) -> int:
        """
        Infer seasonal periods from data frequency.
        
        Args:
            data: Time series data
            
        Returns:
            Inferred seasonal periods
        """
        if isinstance(data.index, pd.DatetimeIndex):
            freq = pd.infer_freq(data.index)
            
            if freq is not None:
                # Map frequency to seasonal periods
                freq_map = {
                    'D': 7,      # Daily -> weekly seasonality
                    'W': 52,     # Weekly -> yearly seasonality
                    'M': 12,     # Monthly -> yearly seasonality
                    'Q': 4,      # Quarterly -> yearly seasonality
                    'H': 24,     # Hourly -> daily seasonality
                }
                
                # Get base frequency (first character)
                base_freq = freq[0] if freq else None
                seasonal_periods = freq_map.get(base_freq, 12)
                
                logger.info(f"Inferred seasonal periods: {seasonal_periods} from frequency: {freq}")
                return seasonal_periods
        
        # Default to 12 if cannot infer
        logger.warning("Could not infer seasonal periods, defaulting to 12")
        return 12
    
    def get_components(self) -> Dict[str, np.ndarray]:
        """
        Get decomposed components (level, trend, seasonal) from fitted model.
        
        Returns:
            Dictionary with component arrays
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        components = {
            'level': self.fitted_model.level,
        }
        
        if self.trend is not None:
            components['trend'] = self.fitted_model.trend
        
        if self.seasonal is not None:
            components['seasonal'] = self.fitted_model.season
        
        return components
    
    def get_residuals(self) -> np.ndarray:
        """
        Get model residuals from fitted model.
        
        Returns:
            Array of residuals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.fitted_model.resid


class SimpleExponentialSmoothing(ETSForecaster):
    """
    Simple Exponential Smoothing (SES) - no trend or seasonal components.
    
    Suitable for data with no clear trend or seasonality.
    """
    
    def __init__(self, name: str = 'SES'):
        """Initialize Simple Exponential Smoothing."""
        super().__init__(
            trend=None,
            seasonal=None,
            seasonal_periods=None,
            damped_trend=False,
            name=name
        )


class HoltLinearTrend(ETSForecaster):
    """
    Holt's Linear Trend method - with trend but no seasonal component.
    
    Suitable for data with trend but no seasonality.
    """
    
    def __init__(self, damped: bool = False, name: str = 'Holt'):
        """
        Initialize Holt's Linear Trend method.
        
        Args:
            damped: Whether to use damped trend
            name: Name identifier
        """
        super().__init__(
            trend='add',
            seasonal=None,
            seasonal_periods=None,
            damped_trend=damped,
            name=name
        )


class HoltWinters(ETSForecaster):
    """
    Holt-Winters method - with both trend and seasonal components.
    
    Suitable for data with both trend and seasonality.
    """
    
    def __init__(
        self,
        seasonal_periods: int = 12,
        trend: str = 'add',
        seasonal: str = 'add',
        damped: bool = False,
        name: str = 'HoltWinters'
    ):
        """
        Initialize Holt-Winters method.
        
        Args:
            seasonal_periods: Number of periods in seasonal cycle
            trend: Type of trend ('add' or 'mul')
            seasonal: Type of seasonal component ('add' or 'mul')
            damped: Whether to use damped trend
            name: Name identifier
        """
        super().__init__(
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=damped,
            name=name
        )


class AutoETS(ETSForecaster):
    """
    Automatic ETS model selection.
    
    Automatically selects the best combination of error, trend, and seasonal components.
    """
    
    def __init__(
        self,
        seasonal_periods: Optional[int] = None,
        auto_seasonal: bool = True,
        information_criterion: str = 'aic',
        name: str = 'AutoETS'
    ):
        """
        Initialize Auto ETS.
        
        Args:
            seasonal_periods: Number of periods in seasonal cycle
            auto_seasonal: Whether to automatically detect seasonality
            information_criterion: Criterion for model selection ('aic' or 'bic')
            name: Name identifier
        """
        super().__init__(name=name)
        self.seasonal_periods = seasonal_periods
        self.auto_seasonal = auto_seasonal
        self.information_criterion = information_criterion
        self.best_config = None
    
    def fit(self, data: pd.Series, **kwargs) -> 'AutoETS':
        """
        Fit ETS model with automatic component selection.
        
        Args:
            data: Historical time series data
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        logger.info("Automatically selecting best ETS configuration")
        
        # Validate data
        data = self.validate_data(data)
        self.training_data = data
        
        # Infer seasonal periods if needed
        if self.seasonal_periods is None and self.auto_seasonal:
            self.seasonal_periods = self._infer_seasonal_periods(data)
        
        # Try different configurations
        configs = self._generate_configurations(data)
        
        best_ic = np.inf
        best_model = None
        best_config = None
        
        for config in configs:
            try:
                model = ExponentialSmoothing(
                    data,
                    trend=config['trend'],
                    seasonal=config['seasonal'],
                    seasonal_periods=config['seasonal_periods'],
                    damped_trend=config['damped_trend']
                )
                
                fitted = model.fit(optimized=True)
                
                # Get information criterion
                ic_value = fitted.aic if self.information_criterion == 'aic' else fitted.bic
                
                if ic_value < best_ic:
                    best_ic = ic_value
                    best_model = fitted
                    best_config = config
                    
            except Exception as e:
                logger.debug(f"Configuration {config} failed: {e}")
                continue
        
        if best_model is None:
            raise ValueError("Failed to fit any ETS configuration")
        
        # Store best model
        self.fitted_model = best_model
        self.best_config = best_config
        self.trend = best_config['trend']
        self.seasonal = best_config['seasonal']
        self.damped_trend = best_config['damped_trend']
        self.is_fitted = True
        
        # Store metadata
        self.training_metadata = {
            'best_config': best_config,
            'aic': best_model.aic,
            'bic': best_model.bic,
            'training_samples': len(data),
            'fit_date': datetime.now().isoformat()
        }
        
        logger.info(
            f"Best configuration selected: trend={best_config['trend']}, "
            f"seasonal={best_config['seasonal']}, damped={best_config['damped_trend']}, "
            f"{self.information_criterion.upper()}={best_ic:.2f}"
        )
        
        return self
    
    def _generate_configurations(self, data: pd.Series) -> list:
        """
        Generate list of ETS configurations to try.
        
        Args:
            data: Time series data
            
        Returns:
            List of configuration dictionaries
        """
        configs = []
        
        # Trend options
        trend_options = [None, 'add']
        if (data > 0).all():
            trend_options.append('mul')
        
        # Seasonal options
        seasonal_options = [None]
        if self.seasonal_periods is not None and len(data) >= 2 * self.seasonal_periods:
            seasonal_options.append('add')
            if (data > 0).all():
                seasonal_options.append('mul')
        
        # Damped options
        damped_options = [False, True]
        
        # Generate all combinations
        for trend in trend_options:
            for seasonal in seasonal_options:
                for damped in damped_options:
                    # Skip damped if no trend
                    if damped and trend is None:
                        continue
                    
                    configs.append({
                        'trend': trend,
                        'seasonal': seasonal,
                        'seasonal_periods': self.seasonal_periods if seasonal else None,
                        'damped_trend': damped
                    })
        
        return configs
