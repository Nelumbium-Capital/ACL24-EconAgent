"""
Backtesting engine for time-series forecast validation.

Implements time-series cross-validation with expanding windows,
computes error metrics (RMSE, MAE), and tests confidence interval coverage.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import json

from src.models.base_forecaster import BaseForecaster
from src.utils.logging_config import logger


class BacktestEngine:
    """
    Backtesting engine for forecast model validation.
    
    Features:
    - Time-series cross-validation with expanding window
    - RMSE, MAE, MAPE computation per model
    - 95% CI coverage testing
    - Calibration reports
    """
    
    def __init__(
        self,
        initial_train_size: int = 36,
        forecast_horizon: int = 1,
        step_size: int = 1,
        confidence_level: float = 0.95,
        target_coverage: float = 0.90,
        output_dir: str = "data/processed/backtest"
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_train_size: Initial training window size (months)
            forecast_horizon: Forecast horizon to evaluate
            step_size: Steps to advance window (1 = monthly)
            confidence_level: Confidence level for prediction intervals
            target_coverage: Target CI coverage (e.g., 0.90 for 90%)
            output_dir: Directory for output files
        """
        self.initial_train_size = initial_train_size
        self.forecast_horizon = forecast_horizon
        self.step_size = step_size
        self.confidence_level = confidence_level
        self.target_coverage = target_coverage
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {}
        
        logger.info(f"Initialized BacktestEngine: train_size={initial_train_size}, horizon={forecast_horizon}")
    
    def backtest_model(
        self,
        model: BaseForecaster,
        data: pd.Series,
        model_name: str = None
    ) -> Dict[str, Any]:
        """
        Backtest a single model using time-series cross-validation.
        
        Args:
            model: Forecaster model to test
            data: Full time series data
            model_name: Name for results (uses model.name if None)
            
        Returns:
            Dictionary with backtest results
        """
        model_name = model_name or model.name
        logger.info(f"Backtesting model: {model_name}")
        
        if len(data) < self.initial_train_size + self.forecast_horizon:
            raise ValueError(f"Insufficient data: need at least {self.initial_train_size + self.forecast_horizon} points")
        
        # Storage for predictions and actuals
        forecasts = []
        actuals = []
        lower_bounds = []
        upper_bounds = []
        train_ends = []
        
        # Expanding window cross-validation
        n_tests = (len(data) - self.initial_train_size - self.forecast_horizon) // self.step_size
        
        logger.info(f"Running {n_tests} cross-validation folds...")
        
        for i in range(0, n_tests * self.step_size, self.step_size):
            train_end = self.initial_train_size + i
            test_start = train_end
            test_end = test_start + self.forecast_horizon
            
            if test_end > len(data):
                break
            
            # Split data
            train_data = data.iloc[:train_end]
            test_data = data.iloc[test_start:test_end]
            
            try:
                # Fit model on training data
                model_instance = self._create_fresh_model(model)
                model_instance.fit(train_data)
                
                # Generate forecast
                result = model_instance.forecast(
                    horizon=self.forecast_horizon,
                    confidence_level=self.confidence_level
                )
                
                # Store results
                forecasts.append(result.point_forecast)
                actuals.append(test_data.values)
                
                if result.lower_bound is not None:
                    lower_bounds.append(result.lower_bound)
                if result.upper_bound is not None:
                    upper_bounds.append(result.upper_bound)
                
                train_ends.append(train_end)
                
            except Exception as e:
                logger.warning(f"Fold {i} failed: {e}")
                continue
        
        # Convert to arrays
        forecasts = np.array(forecasts)
        actuals = np.array(actuals)
        
        # Compute error metrics
        metrics = self._compute_metrics(forecasts, actuals)
        
        # Test CI coverage if bounds available
        coverage_metrics = {}
        if lower_bounds and upper_bounds:
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)
            coverage_metrics = self._test_ci_coverage(actuals, lower_bounds, upper_bounds)
        
        # Compile results
        results = {
            'model_name': model_name,
            'n_folds': len(forecasts),
            'metrics': metrics,
            'coverage': coverage_metrics,
            'forecasts': forecasts.tolist(),
            'actuals': actuals.tolist(),
            'train_ends': train_ends
        }
        
        self.results[model_name] = results
        
        # Log summary
        logger.info(f"Backtest complete for {model_name}:")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        if coverage_metrics:
            logger.info(f"  CI Coverage: {coverage_metrics['coverage']:.2%} (target: {self.target_coverage:.2%})")
        
        return results
    
    def backtest_multiple_models(
        self,
        models: List[Tuple[BaseForecaster, str]],
        data: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Backtest multiple models on the same data.
        
        Args:
            models: List of (model, name) tuples
            data: Time series data
            
        Returns:
            Dictionary mapping model name to results
        """
        logger.info(f"Backtesting {len(models)} models...")
        
        all_results = {}
        
        for model, model_name in models:
            try:
                results = self.backtest_model(model, data, model_name)
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Failed to backtest {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}
        
        # Generate comparison report
        self._generate_comparison_report(all_results)
        
        return all_results
    
    def _create_fresh_model(self, model: BaseForecaster) -> BaseForecaster:
        """Create a fresh instance of the model."""
        import copy
        try:
            # Try deep copy first
            return copy.deepcopy(model)
        except:
            # Fallback: create new instance with same class
            return model.__class__(name=model.name)
    
    def _compute_metrics(
        self,
        forecasts: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute error metrics.
        
        Args:
            forecasts: Predicted values
            actuals: Actual values
            
        Returns:
            Dictionary of metrics
        """
        # Flatten if multi-step forecasts
        if forecasts.ndim > 1:
            forecasts = forecasts.flatten()
            actuals = actuals.flatten()
        
        # Remove any NaN values
        mask = ~(np.isnan(forecasts) | np.isnan(actuals))
        forecasts = forecasts[mask]
        actuals = actuals[mask]
        
        if len(forecasts) == 0:
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'mse': np.nan,
                'mean_error': np.nan
            }
        
        # Compute metrics
        errors = forecasts - actuals
        squared_errors = errors ** 2
        absolute_errors = np.abs(errors)
        
        metrics = {
            'rmse': np.sqrt(np.mean(squared_errors)),
            'mae': np.mean(absolute_errors),
            'mape': np.mean(np.abs(errors / (actuals + 1e-8))) * 100,  # Percentage
            'mse': np.mean(squared_errors),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_ae': np.median(absolute_errors),
            'max_ae': np.max(absolute_errors)
        }
        
        return metrics
    
    def _test_ci_coverage(
        self,
        actuals: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray
    ) -> Dict[str, float]:
        """
        Test confidence interval coverage.
        
        Args:
            actuals: Actual values
            lower_bounds: Lower CI bounds
            upper_bounds: Upper CI bounds
            
        Returns:
            Dictionary with coverage metrics
        """
        # Flatten arrays
        if actuals.ndim > 1:
            actuals = actuals.flatten()
            lower_bounds = lower_bounds.flatten()
            upper_bounds = upper_bounds.flatten()
        
        # Check coverage
        within_ci = (actuals >= lower_bounds) & (actuals <= upper_bounds)
        coverage = np.mean(within_ci)
        
        # Check if coverage meets target
        meets_target = coverage >= self.target_coverage
        
        # Compute average interval width
        avg_width = np.mean(upper_bounds - lower_bounds)
        
        metrics = {
            'coverage': coverage,
            'target_coverage': self.target_coverage,
            'meets_target': bool(meets_target),
            'avg_interval_width': avg_width,
            'n_observations': len(actuals),
            'n_within_ci': int(np.sum(within_ci))
        }
        
        return metrics
    
    def _generate_comparison_report(self, all_results: Dict[str, Dict[str, Any]]):
        """Generate comparison report across models."""
        logger.info("\n" + "="*70)
        logger.info("BACKTEST COMPARISON REPORT")
        logger.info("="*70)
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name, results in all_results.items():
            if 'error' in results:
                continue
            
            metrics = results['metrics']
            coverage = results.get('coverage', {})
            
            row = {
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'MAPE': metrics['mape'],
                'Mean Error': metrics['mean_error'],
                'N Folds': results['n_folds']
            }
            
            if coverage:
                row['CI Coverage'] = coverage['coverage']
                row['Meets Target'] = 'Yes' if coverage['meets_target'] else 'No'
            
            comparison_data.append(row)
        
        if not comparison_data:
            logger.warning("No successful backtest results to compare")
            return
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by RMSE
        comparison_df = comparison_df.sort_values('RMSE')
        
        # Display
        logger.info("\nModel Comparison (sorted by RMSE):")
        logger.info(comparison_df.to_string(index=False))
        
        # Save to file
        output_file = self.output_dir / "backtest_comparison.csv"
        comparison_df.to_csv(output_file, index=False)
        logger.info(f"\n✓ Comparison saved to {output_file}")
        
        # Save detailed JSON
        json_file = self.output_dir / "backtest_results.json"
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"✓ Detailed results saved to {json_file}")
    
    def generate_calibration_report(self, model_name: str) -> str:
        """
        Generate detailed calibration report for a model.
        
        Args:
            model_name: Name of model
            
        Returns:
            Report text
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found in results")
        
        results = self.results[model_name]
        metrics = results['metrics']
        coverage = results.get('coverage', {})
        
        report = f"""
CALIBRATION REPORT: {model_name}
{'='*70}

FORECAST ACCURACY:
  RMSE:        {metrics['rmse']:.4f}
  MAE:         {metrics['mae']:.4f}
  MAPE:        {metrics['mape']:.2f}%
  Mean Error:  {metrics['mean_error']:.4f}
  Std Error:   {metrics['std_error']:.4f}
  
CROSS-VALIDATION:
  Number of folds: {results['n_folds']}
  Horizon: {self.forecast_horizon}
  Initial train size: {self.initial_train_size}
"""
        
        if coverage:
            report += f"""
CONFIDENCE INTERVAL CALIBRATION:
  Coverage:        {coverage['coverage']:.2%}
  Target:          {coverage['target_coverage']:.2%}
  Meets Target:    {'✓ Yes' if coverage['meets_target'] else '✗ No'}
  Avg Width:       {coverage['avg_interval_width']:.4f}
  Within CI:       {coverage['n_within_ci']} / {coverage['n_observations']}
"""
        
        report += f"\n{'='*70}\n"
        
        # Save report
        report_file = self.output_dir / f"calibration_report_{model_name}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"✓ Calibration report saved to {report_file}")
        
        return report


def main():
    """Run backtest demo."""
    from src.models.arima_forecaster import ARIMAForecaster
    from src.models.ets_forecaster import ETSForecaster
    from src.data.fred_client import FREDClient
    
    logger.info("Starting Backtest Engine Demo")
    
    # Fetch data
    client = FREDClient()
    data = client.fetch_series('UNRATE', '2018-01-01', '2024-01-01')
    
    if data is None or len(data) < 50:
        logger.error("Failed to fetch data")
        return
    
    logger.info(f"Loaded {len(data)} data points")
    
    # Create models
    models = [
        (ARIMAForecaster(auto_order=True, name='ARIMA'), 'ARIMA'),
        (ETSForecaster(trend='add', seasonal=None, name='ETS'), 'ETS')
    ]
    
    # Run backtest
    engine = BacktestEngine(
        initial_train_size=36,
        forecast_horizon=1,
        confidence_level=0.95,
        target_coverage=0.90
    )
    
    results = engine.backtest_multiple_models(models, data)
    
    # Generate calibration reports
    for model_name in results.keys():
        if 'error' not in results[model_name]:
            report = engine.generate_calibration_report(model_name)
            print(report)
    
    logger.info("\n✓ Backtest Demo Complete")


if __name__ == "__main__":
    main()


