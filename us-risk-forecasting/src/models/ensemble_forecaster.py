"""
Ensemble forecasting model that combines predictions from multiple models.

Implements weighted averaging with dynamic weight optimization based on
validation performance and recent accuracy.
"""
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.base_forecaster import BaseForecaster, ForecastResult
from src.utils.logging_config import logger


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble forecaster that combines multiple models using weighted averaging.
    
    Supports:
    - Configurable weights for each model
    - Automatic weight optimization based on validation performance
    - Dynamic weight adjustment based on recent accuracy
    """
    
    def __init__(
        self,
        models: List[BaseForecaster],
        weights: Optional[np.ndarray] = None,
        weight_optimization: str = 'equal',
        recent_window: int = 12,
        name: str = 'Ensemble'
    ):
        """
        Initialize ensemble forecaster.
        
        Args:
            models: List of fitted forecaster models
            weights: Optional initial weights for each model (must sum to 1)
            weight_optimization: Method for weight optimization
                - 'equal': Equal weights for all models
                - 'inverse_error': Weights inversely proportional to validation error
                - 'optimize': Optimize weights to minimize validation error
                - 'dynamic': Adjust weights based on recent performance
            recent_window: Number of recent periods for dynamic weighting
            name: Name identifier for this forecaster
        """
        super().__init__(name)
        
        if not models:
            raise ValueError("At least one model must be provided")
        
        self.models = models
        self.n_models = len(models)
        self.weight_optimization = weight_optimization
        self.recent_window = recent_window
        
        # Initialize weights
        if weights is not None:
            if len(weights) != self.n_models:
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({self.n_models})")
            if not np.isclose(np.sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
            self.weights = np.array(weights)
        else:
            # Default to equal weights
            self.weights = np.ones(self.n_models) / self.n_models
        
        self.validation_errors = None
        self.recent_errors = None
        
        logger.info(f"Initialized ensemble with {self.n_models} models")
        logger.info(f"Model names: {[m.name for m in self.models]}")
        logger.info(f"Initial weights: {self.weights}")
    
    def fit(self, data: pd.Series, validation_data: Optional[pd.Series] = None, **kwargs) -> 'EnsembleForecaster':
        """
        Fit ensemble by optimizing weights based on validation performance.
        
        Args:
            data: Historical time series data for training
            validation_data: Optional validation data for weight optimization
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.name} ensemble")
        
        # Validate data
        data = self.validate_data(data)
        self.training_data = data
        
        # Fit all models if not already fitted
        for i, model in enumerate(self.models):
            if not model.is_fitted:
                logger.info(f"Fitting model {i+1}/{self.n_models}: {model.name}")
                model.fit(data)
        
        # Optimize weights if validation data provided
        if validation_data is not None:
            validation_data = self.validate_data(validation_data)
            self._optimize_weights(validation_data)
        elif self.weight_optimization != 'equal':
            logger.warning("No validation data provided, using equal weights")
            self.weight_optimization = 'equal'
            self.weights = np.ones(self.n_models) / self.n_models
        
        self.is_fitted = True
        
        # Store metadata
        self.training_metadata = {
            'n_models': self.n_models,
            'model_names': [m.name for m in self.models],
            'weights': self.weights.tolist(),
            'weight_optimization': self.weight_optimization,
            'training_samples': len(data),
            'fit_date': datetime.now().isoformat()
        }
        
        logger.info(f"Ensemble fitted successfully with weights: {self.weights}")
        
        return self
    
    def _optimize_weights(self, validation_data: pd.Series):
        """
        Optimize ensemble weights based on validation performance.
        
        Args:
            validation_data: Validation time series data
        """
        logger.info(f"Optimizing weights using method: {self.weight_optimization}")
        
        # Generate forecasts from all models
        horizon = min(len(validation_data), 12)  # Use up to 12 steps for validation
        forecasts = []
        errors = []
        
        for model in self.models:
            try:
                # Get forecast
                if hasattr(model, 'forecast'):
                    result = model.forecast(horizon=horizon)
                    if isinstance(result, dict):
                        # Handle multivariate models (like DeepVAR)
                        # Use first variable for weight optimization
                        first_var = list(result.keys())[0]
                        pred = result[first_var].point_forecast
                    else:
                        pred = result.point_forecast
                else:
                    logger.warning(f"Model {model.name} does not have forecast method")
                    pred = np.zeros(horizon)
                
                forecasts.append(pred[:horizon])
                
                # Calculate error
                actual = validation_data.values[:horizon]
                error = np.sqrt(np.mean((pred[:horizon] - actual) ** 2))
                errors.append(error)
                
            except Exception as e:
                logger.error(f"Error generating forecast for {model.name}: {e}")
                forecasts.append(np.zeros(horizon))
                errors.append(float('inf'))
        
        self.validation_errors = np.array(errors)
        forecasts = np.array(forecasts)
        
        logger.info(f"Validation errors: {self.validation_errors}")
        
        # Optimize weights based on method
        if self.weight_optimization == 'equal':
            self.weights = np.ones(self.n_models) / self.n_models
            
        elif self.weight_optimization == 'inverse_error':
            # Weights inversely proportional to error
            # Avoid division by zero
            inv_errors = 1.0 / (self.validation_errors + 1e-8)
            self.weights = inv_errors / np.sum(inv_errors)
            
        elif self.weight_optimization == 'optimize':
            # Optimize weights to minimize ensemble error
            actual = validation_data.values[:horizon]
            
            def objective(w):
                ensemble_pred = np.average(forecasts, axis=0, weights=w)
                return np.sqrt(np.mean((ensemble_pred - actual) ** 2))
            
            # Constraints: weights sum to 1
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(self.n_models)]
            
            # Initial guess
            x0 = self.weights
            
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                self.weights = result.x
                logger.info(f"Optimization successful. Final error: {result.fun:.6f}")
            else:
                logger.warning(f"Optimization failed: {result.message}")
                # Fall back to inverse error weighting
                inv_errors = 1.0 / (self.validation_errors + 1e-8)
                self.weights = inv_errors / np.sum(inv_errors)
        
        elif self.weight_optimization == 'dynamic':
            # Start with inverse error weighting
            inv_errors = 1.0 / (self.validation_errors + 1e-8)
            self.weights = inv_errors / np.sum(inv_errors)
            # Dynamic adjustment will happen during forecasting
        
        logger.info(f"Optimized weights: {self.weights}")
    
    def forecast(
        self,
        horizon: int,
        confidence_level: float = 0.95,
        return_individual: bool = False
    ) -> ForecastResult:
        """
        Generate ensemble forecast by combining individual model predictions.
        
        Args:
            horizon: Number of steps to forecast
            confidence_level: Confidence level for prediction intervals
            return_individual: Whether to include individual model forecasts in metadata
            
        Returns:
            ForecastResult with ensemble forecast
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before forecasting")
        
        logger.info(f"Generating {horizon}-step ensemble forecast")
        
        # Generate forecasts from all models
        forecasts = []
        individual_results = []
        
        for i, model in enumerate(self.models):
            try:
                result = model.forecast(horizon=horizon, confidence_level=confidence_level)
                
                if isinstance(result, dict):
                    # Handle multivariate models
                    first_var = list(result.keys())[0]
                    pred = result[first_var].point_forecast
                else:
                    pred = result.point_forecast
                
                forecasts.append(pred)
                individual_results.append(result)
                
                logger.info(f"Model {model.name} forecast: mean={np.mean(pred):.4f}")
                
            except Exception as e:
                logger.error(f"Error generating forecast for {model.name}: {e}")
                # Use zeros as fallback
                forecasts.append(np.zeros(horizon))
                individual_results.append(None)
        
        forecasts = np.array(forecasts)
        
        # Adjust weights dynamically if enabled
        if self.weight_optimization == 'dynamic' and self.recent_errors is not None:
            self._adjust_weights_dynamically()
        
        # Combine forecasts using weighted average
        ensemble_forecast = np.average(forecasts, axis=0, weights=self.weights)
        
        # Calculate ensemble prediction intervals
        # Use weighted combination of individual intervals
        lower_bounds = []
        upper_bounds = []
        
        for result in individual_results:
            if result is not None:
                if isinstance(result, dict):
                    first_var = list(result.keys())[0]
                    result = result[first_var]
                
                if result.lower_bound is not None:
                    lower_bounds.append(result.lower_bound)
                if result.upper_bound is not None:
                    upper_bounds.append(result.upper_bound)
        
        if lower_bounds and upper_bounds:
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
            
            ensemble_lower = np.average(lower_bounds, axis=0, weights=self.weights)
            ensemble_upper = np.average(upper_bounds, axis=0, weights=self.weights)
        else:
            # Fallback: calculate intervals from forecast variance
            forecast_std = np.std(forecasts, axis=0)
            from scipy import stats
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            ensemble_lower = ensemble_forecast - z_score * forecast_std
            ensemble_upper = ensemble_forecast + z_score * forecast_std
        
        # Create metadata
        metadata = {
            'n_models': self.n_models,
            'model_names': [m.name for m in self.models],
            'weights': self.weights.tolist(),
            'weight_optimization': self.weight_optimization
        }
        
        if return_individual:
            metadata['individual_forecasts'] = {
                model.name: forecasts[i].tolist()
                for i, model in enumerate(self.models)
            }
        
        # Create result
        result = ForecastResult(
            model_name=self.name,
            series_name=self.training_data.name if self.training_data.name else 'series',
            forecast_date=datetime.now(),
            horizon=horizon,
            point_forecast=ensemble_forecast,
            lower_bound=ensemble_lower,
            upper_bound=ensemble_upper,
            confidence_level=confidence_level,
            metadata=metadata
        )
        
        logger.info(f"Ensemble forecast generated successfully")
        logger.info(f"Forecast range: [{np.min(ensemble_forecast):.4f}, {np.max(ensemble_forecast):.4f}]")
        
        return result
    
    def _adjust_weights_dynamically(self):
        """
        Adjust weights based on recent forecast accuracy.
        
        Uses exponential weighting to give more importance to recent performance.
        """
        if self.recent_errors is None or len(self.recent_errors) == 0:
            return
        
        # Calculate recent performance for each model
        recent_rmse = np.sqrt(np.mean(self.recent_errors ** 2, axis=0))
        
        # Inverse error weighting with recent performance
        inv_errors = 1.0 / (recent_rmse + 1e-8)
        new_weights = inv_errors / np.sum(inv_errors)
        
        # Smooth transition: blend old and new weights
        alpha = 0.3  # Weight for new weights
        self.weights = alpha * new_weights + (1 - alpha) * self.weights
        
        # Normalize
        self.weights = self.weights / np.sum(self.weights)
        
        logger.info(f"Dynamically adjusted weights: {self.weights}")
    
    def update_recent_errors(self, actual: np.ndarray, forecasts: Dict[str, np.ndarray]):
        """
        Update recent error tracking for dynamic weighting.
        
        Args:
            actual: Actual observed values
            forecasts: Dictionary mapping model names to their forecasts
        """
        errors = []
        
        for model in self.models:
            if model.name in forecasts:
                pred = forecasts[model.name]
                error = actual - pred
                errors.append(error)
            else:
                errors.append(np.zeros_like(actual))
        
        errors = np.array(errors).T  # Shape: (n_periods, n_models)
        
        if self.recent_errors is None:
            self.recent_errors = errors
        else:
            # Append and keep only recent window
            self.recent_errors = np.vstack([self.recent_errors, errors])
            if len(self.recent_errors) > self.recent_window:
                self.recent_errors = self.recent_errors[-self.recent_window:]
        
        logger.info(f"Updated recent errors. Window size: {len(self.recent_errors)}")
    
    def get_model_contributions(self, horizon: int = 1) -> pd.DataFrame:
        """
        Get contribution of each model to the ensemble forecast.
        
        Args:
            horizon: Forecast horizon
            
        Returns:
            DataFrame with model contributions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
        
        # Generate forecasts
        forecasts = []
        
        for model in self.models:
            try:
                result = model.forecast(horizon=horizon)
                if isinstance(result, dict):
                    first_var = list(result.keys())[0]
                    pred = result[first_var].point_forecast
                else:
                    pred = result.point_forecast
                forecasts.append(pred)
            except Exception as e:
                logger.error(f"Error getting forecast for {model.name}: {e}")
                forecasts.append(np.zeros(horizon))
        
        forecasts = np.array(forecasts)
        
        # Calculate contributions
        contributions = forecasts * self.weights[:, np.newaxis]
        
        # Create DataFrame
        df = pd.DataFrame(
            contributions.T,
            columns=[m.name for m in self.models]
        )
        df['Ensemble'] = df.sum(axis=1)
        
        return df
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the ensemble and its models.
        
        Returns:
            Dictionary with ensemble information
        """
        info = super().get_model_info()
        
        info['models'] = [
            {
                'name': model.name,
                'weight': float(self.weights[i]),
                'is_fitted': model.is_fitted,
                'validation_error': float(self.validation_errors[i]) if self.validation_errors is not None else None
            }
            for i, model in enumerate(self.models)
        ]
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the ensemble."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        model_names = ", ".join([m.name for m in self.models])
        return f"{self.__class__.__name__}(name='{self.name}', models=[{model_names}], {fitted_status})"
