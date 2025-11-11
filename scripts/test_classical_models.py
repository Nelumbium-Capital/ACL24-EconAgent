"""
Test script for classical time-series forecasting models.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models import (
    ARIMAForecaster,
    SARIMAForecaster,
    SimpleExponentialSmoothing,
    HoltLinearTrend,
    HoltWinters,
    AutoETS
)


def generate_test_data(n_points: int = 100, trend: bool = True, seasonal: bool = False) -> pd.Series:
    """Generate synthetic time series data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='M')
    
    # Base level
    data = np.ones(n_points) * 100
    
    # Add trend
    if trend:
        data += np.linspace(0, 20, n_points)
    
    # Add seasonality
    if seasonal:
        seasonal_pattern = 10 * np.sin(np.linspace(0, 8*np.pi, n_points))
        data += seasonal_pattern
    
    # Add noise
    noise = np.random.normal(0, 2, n_points)
    data += noise
    
    return pd.Series(data, index=dates, name='test_series')


def test_arima():
    """Test ARIMA forecaster."""
    print("\n" + "="*60)
    print("Testing ARIMA Forecaster")
    print("="*60)
    
    # Generate test data
    data = generate_test_data(n_points=60, trend=True, seasonal=False)
    print(f"Generated {len(data)} data points")
    
    # Create and fit model
    model = ARIMAForecaster(auto_order=True)
    model.fit(data)
    
    print(f"Model fitted: {model}")
    print(f"Order: {model.order}")
    print(f"AIC: {model.fitted_model.aic:.2f}")
    print(f"BIC: {model.fitted_model.bic:.2f}")
    
    # Generate forecast
    forecast_result = model.forecast(horizon=12)
    
    print(f"\nForecast generated:")
    print(f"  Horizon: {forecast_result.horizon}")
    print(f"  Point forecast (first 5): {forecast_result.point_forecast[:5]}")
    print(f"  Lower bound (first 5): {forecast_result.lower_bound[:5]}")
    print(f"  Upper bound (first 5): {forecast_result.upper_bound[:5]}")
    
    return True


def test_sarima():
    """Test SARIMA forecaster."""
    print("\n" + "="*60)
    print("Testing SARIMA Forecaster")
    print("="*60)
    
    # Generate test data with seasonality
    data = generate_test_data(n_points=72, trend=True, seasonal=True)
    print(f"Generated {len(data)} data points with seasonality")
    
    # Create and fit model
    model = SARIMAForecaster(seasonal_period=12, auto_order=True)
    model.fit(data)
    
    print(f"Model fitted: {model}")
    print(f"Order: {model.order}")
    print(f"Seasonal order: {model.seasonal_order}")
    print(f"AIC: {model.fitted_model.aic:.2f}")
    print(f"BIC: {model.fitted_model.bic:.2f}")
    
    # Generate forecast
    forecast_result = model.forecast(horizon=12)
    
    print(f"\nForecast generated:")
    print(f"  Horizon: {forecast_result.horizon}")
    print(f"  Point forecast (first 5): {forecast_result.point_forecast[:5]}")
    
    return True


def test_simple_exponential_smoothing():
    """Test Simple Exponential Smoothing."""
    print("\n" + "="*60)
    print("Testing Simple Exponential Smoothing")
    print("="*60)
    
    # Generate test data without trend or seasonality
    data = generate_test_data(n_points=50, trend=False, seasonal=False)
    print(f"Generated {len(data)} data points")
    
    # Create and fit model
    model = SimpleExponentialSmoothing()
    model.fit(data)
    
    print(f"Model fitted: {model}")
    print(f"AIC: {model.fitted_model.aic:.2f}")
    print(f"BIC: {model.fitted_model.bic:.2f}")
    
    # Generate forecast
    forecast_result = model.forecast(horizon=12)
    
    print(f"\nForecast generated:")
    print(f"  Horizon: {forecast_result.horizon}")
    print(f"  Point forecast (first 5): {forecast_result.point_forecast[:5]}")
    
    return True


def test_holt_linear_trend():
    """Test Holt's Linear Trend method."""
    print("\n" + "="*60)
    print("Testing Holt's Linear Trend")
    print("="*60)
    
    # Generate test data with trend
    data = generate_test_data(n_points=50, trend=True, seasonal=False)
    print(f"Generated {len(data)} data points with trend")
    
    # Create and fit model
    model = HoltLinearTrend(damped=False)
    model.fit(data)
    
    print(f"Model fitted: {model}")
    print(f"AIC: {model.fitted_model.aic:.2f}")
    print(f"BIC: {model.fitted_model.bic:.2f}")
    
    # Generate forecast
    forecast_result = model.forecast(horizon=12)
    
    print(f"\nForecast generated:")
    print(f"  Horizon: {forecast_result.horizon}")
    print(f"  Point forecast (first 5): {forecast_result.point_forecast[:5]}")
    
    return True


def test_holt_winters():
    """Test Holt-Winters method."""
    print("\n" + "="*60)
    print("Testing Holt-Winters")
    print("="*60)
    
    # Generate test data with trend and seasonality
    data = generate_test_data(n_points=72, trend=True, seasonal=True)
    print(f"Generated {len(data)} data points with trend and seasonality")
    
    # Create and fit model
    model = HoltWinters(seasonal_periods=12, trend='add', seasonal='add')
    model.fit(data)
    
    print(f"Model fitted: {model}")
    print(f"AIC: {model.fitted_model.aic:.2f}")
    print(f"BIC: {model.fitted_model.bic:.2f}")
    
    # Generate forecast
    forecast_result = model.forecast(horizon=12)
    
    print(f"\nForecast generated:")
    print(f"  Horizon: {forecast_result.horizon}")
    print(f"  Point forecast (first 5): {forecast_result.point_forecast[:5]}")
    
    return True


def test_auto_ets():
    """Test Auto ETS."""
    print("\n" + "="*60)
    print("Testing Auto ETS")
    print("="*60)
    
    # Generate test data
    data = generate_test_data(n_points=72, trend=True, seasonal=True)
    print(f"Generated {len(data)} data points")
    
    # Create and fit model
    model = AutoETS(seasonal_periods=12, auto_seasonal=True)
    model.fit(data)
    
    print(f"Model fitted: {model}")
    print(f"Best configuration: {model.best_config}")
    print(f"AIC: {model.fitted_model.aic:.2f}")
    print(f"BIC: {model.fitted_model.bic:.2f}")
    
    # Generate forecast
    forecast_result = model.forecast(horizon=12)
    
    print(f"\nForecast generated:")
    print(f"  Horizon: {forecast_result.horizon}")
    print(f"  Point forecast (first 5): {forecast_result.point_forecast[:5]}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Classical Time-Series Forecasting Models Test Suite")
    print("="*60)
    
    tests = [
        ("ARIMA", test_arima),
        ("SARIMA", test_sarima),
        ("Simple Exponential Smoothing", test_simple_exponential_smoothing),
        ("Holt Linear Trend", test_holt_linear_trend),
        ("Holt-Winters", test_holt_winters),
        ("Auto ETS", test_auto_ets)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            results[test_name] = f"ERROR: {str(e)}"
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    # Overall result
    all_passed = all(r == "PASSED" for r in results.values())
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
