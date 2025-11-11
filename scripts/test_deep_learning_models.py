"""
Test script for deep learning forecasting models.

Tests the DeepVAR, LSTM, and Ensemble forecasters with synthetic data.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models import (
    DeepVARForecaster,
    LSTMForecaster,
    ARIMAForecaster,
    EnsembleForecaster
)
from src.utils.logging_config import logger


def generate_synthetic_data(n_periods: int = 200, n_variables: int = 3) -> pd.DataFrame:
    """Generate synthetic multivariate time series data."""
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='M')
    
    # Generate correlated time series
    np.random.seed(42)
    t = np.arange(n_periods)
    
    # Trend + seasonality + noise
    data = {}
    for i in range(n_variables):
        trend = 0.1 * t
        seasonal = 5 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 1, n_periods)
        data[f'var_{i+1}'] = 50 + trend + seasonal + noise
    
    df = pd.DataFrame(data, index=dates)
    return df


def test_deep_var():
    """Test Deep VAR forecaster."""
    logger.info("=" * 60)
    logger.info("Testing Deep VAR Forecaster")
    logger.info("=" * 60)
    
    # Generate data
    data = generate_synthetic_data(n_periods=150, n_variables=3)
    logger.info(f"Generated data shape: {data.shape}")
    
    # Initialize and fit model
    model = DeepVARForecaster(
        lag_order=6,
        hidden_dims=[32, 16],
        epochs=20,
        batch_size=16,
        early_stopping_patience=5
    )
    
    logger.info("Fitting Deep VAR model...")
    model.fit(data)
    
    # Generate forecast
    horizon = 12
    logger.info(f"Generating {horizon}-step forecast...")
    forecasts = model.forecast(horizon=horizon)
    
    # Display results
    for var_name, result in forecasts.items():
        logger.info(f"\n{var_name} forecast:")
        logger.info(f"  Mean: {np.mean(result.point_forecast):.2f}")
        logger.info(f"  Range: [{np.min(result.point_forecast):.2f}, {np.max(result.point_forecast):.2f}]")
    
    logger.info("\n✓ Deep VAR test passed!")
    return model


def test_lstm():
    """Test LSTM forecaster."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing LSTM Forecaster")
    logger.info("=" * 60)
    
    # Generate univariate data
    data = generate_synthetic_data(n_periods=150, n_variables=1)
    series = data.iloc[:, 0]
    series.name = 'test_series'
    logger.info(f"Generated data length: {len(series)}")
    
    # Initialize and fit model
    model = LSTMForecaster(
        lookback_window=12,
        hidden_dim=32,
        num_layers=2,
        epochs=20,
        batch_size=16,
        early_stopping_patience=5
    )
    
    logger.info("Fitting LSTM model...")
    model.fit(series)
    
    # Generate forecast
    horizon = 12
    logger.info(f"Generating {horizon}-step forecast...")
    result = model.forecast(horizon=horizon)
    
    # Display results
    logger.info(f"\nForecast results:")
    logger.info(f"  Mean: {np.mean(result.point_forecast):.2f}")
    logger.info(f"  Range: [{np.min(result.point_forecast):.2f}, {np.max(result.point_forecast):.2f}]")
    logger.info(f"  Confidence intervals: [{np.mean(result.lower_bound):.2f}, {np.mean(result.upper_bound):.2f}]")
    
    logger.info("\n✓ LSTM test passed!")
    return model


def test_ensemble():
    """Test Ensemble forecaster."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Ensemble Forecaster")
    logger.info("=" * 60)
    
    # Generate data
    data = generate_synthetic_data(n_periods=150, n_variables=1)
    series = data.iloc[:, 0]
    series.name = 'test_series'
    
    # Split into train and validation
    train_size = int(len(series) * 0.8)
    train_data = series[:train_size]
    val_data = series[train_size:]
    
    logger.info(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")
    
    # Create individual models
    logger.info("Creating individual models...")
    
    arima_model = ARIMAForecaster(order=(1, 1, 1), auto_order=False, name='ARIMA')
    lstm_model = LSTMForecaster(
        lookback_window=12,
        hidden_dim=16,
        num_layers=1,
        epochs=10,
        batch_size=16,
        name='LSTM'
    )
    
    models = [arima_model, lstm_model]
    
    # Create ensemble
    logger.info("Creating ensemble...")
    ensemble = EnsembleForecaster(
        models=models,
        weight_optimization='optimize',
        name='Ensemble'
    )
    
    # Fit ensemble
    logger.info("Fitting ensemble...")
    ensemble.fit(train_data, validation_data=val_data)
    
    # Generate forecast
    horizon = 12
    logger.info(f"Generating {horizon}-step forecast...")
    result = ensemble.forecast(horizon=horizon, return_individual=True)
    
    # Display results
    logger.info(f"\nEnsemble forecast results:")
    logger.info(f"  Mean: {np.mean(result.point_forecast):.2f}")
    logger.info(f"  Range: [{np.min(result.point_forecast):.2f}, {np.max(result.point_forecast):.2f}]")
    logger.info(f"  Model weights: {ensemble.weights}")
    
    # Show model contributions
    contributions = ensemble.get_model_contributions(horizon=horizon)
    logger.info(f"\nModel contributions:")
    logger.info(f"\n{contributions.head()}")
    
    logger.info("\n✓ Ensemble test passed!")
    return ensemble


def main():
    """Run all tests."""
    logger.info("Starting deep learning models test suite")
    logger.info("=" * 60)
    
    try:
        # Test Deep VAR
        deep_var_model = test_deep_var()
        
        # Test LSTM
        lstm_model = test_lstm()
        
        # Test Ensemble
        ensemble_model = test_ensemble()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED! ✓")
        logger.info("=" * 60)
        logger.info("\nSummary:")
        logger.info(f"  - Deep VAR: {deep_var_model.n_variables} variables, {deep_var_model.training_metadata['epochs_trained']} epochs")
        logger.info(f"  - LSTM: {lstm_model.training_metadata['epochs_trained']} epochs")
        logger.info(f"  - Ensemble: {ensemble_model.n_models} models combined")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
