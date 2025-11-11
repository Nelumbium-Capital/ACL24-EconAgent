"""
Test script for calibration engine functionality.

Tests:
- Backtesting with time-series cross-validation
- Performance metrics calculation
- Hyperparameter optimization
- Model comparison and selection
- Model versioning and rollback
- Automated retraining
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
    ETSForecaster,
    CalibrationEngine,
    AutoRetrainingEngine,
    RetrainingConfig
)
from src.utils.logging_config import logger


def generate_test_data(n_points: int = 200) -> pd.Series:
    """Generate synthetic time series data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='M')
    
    # Generate data with trend and seasonality
    trend = np.linspace(100, 150, n_points)
    seasonal = 10 * np.sin(np.arange(n_points) * 2 * np.pi / 12)
    noise = np.random.normal(0, 5, n_points)
    
    values = trend + seasonal + noise
    
    return pd.Series(values, index=dates, name='test_series')


def test_backtesting():
    """Test backtesting functionality."""
    logger.info("=" * 60)
    logger.info("TEST 1: Backtesting")
    logger.info("=" * 60)
    
    # Generate test data
    data = generate_test_data(150)
    
    # Create models
    models = [
        ARIMAForecaster(name='ARIMA_auto', auto_order=True),
        ETSForecaster(name='ETS_auto', auto_model=True)
    ]
    
    # Create calibration engine
    engine = CalibrationEngine(models=models)
    
    # Run backtest
    logger.info("Running backtest with 5 folds...")
    results = engine.backtest(
        data=data,
        n_splits=5,
        horizon=12,
        expanding_window=True
    )
    
    # Display results
    for model_name, result in results.items():
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Average MAE: {result.average_metrics['mae']:.4f}")
        logger.info(f"  Average RMSE: {result.average_metrics['rmse']:.4f}")
        logger.info(f"  Average MAPE: {result.average_metrics['mape']:.2f}%")
        logger.info(f"  Directional Accuracy: {result.average_metrics['directional_accuracy']:.2%}")
        logger.info(f"  Number of folds: {len(result.fold_results)}")
    
    logger.info("\n✓ Backtesting test passed")
    return engine, results


def test_model_comparison(engine, backtest_results):
    """Test model comparison functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Model Comparison")
    logger.info("=" * 60)
    
    # Compare models
    comparison = engine.compare_models(backtest_results, metric='rmse')
    
    logger.info("\nModel Comparison (sorted by RMSE):")
    logger.info(f"\n{comparison.to_string()}")
    
    # Select best model
    best_model = engine.select_best_model(backtest_results, metric='rmse')
    logger.info(f"\nBest model: {best_model}")
    
    logger.info("\n✓ Model comparison test passed")
    return best_model


def test_hyperparameter_optimization():
    """Test hyperparameter optimization."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Hyperparameter Optimization")
    logger.info("=" * 60)
    
    # Generate test data
    data = generate_test_data(100)
    
    # Create calibration engine
    engine = CalibrationEngine()
    
    # Define parameter grid (small for testing)
    param_grid = {
        'order': [(1, 1, 1), (2, 1, 1)],
        'auto_order': [False],
        'name': ['ARIMA_optimized']
    }
    
    logger.info("Running hyperparameter optimization...")
    logger.info(f"Testing {len(param_grid['order'])} parameter combinations")
    
    best_params, best_score = engine.optimize_hyperparameters(
        model_class=ARIMAForecaster,
        data=data,
        param_grid=param_grid,
        n_splits=3,
        horizon=6,
        metric='rmse'
    )
    
    logger.info(f"\nBest parameters: {best_params}")
    logger.info(f"Best RMSE: {best_score:.4f}")
    
    logger.info("\n✓ Hyperparameter optimization test passed")
    return best_params


def test_model_versioning():
    """Test model versioning and rollback."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Model Versioning and Rollback")
    logger.info("=" * 60)
    
    # Generate test data
    data = generate_test_data(100)
    
    # Create calibration engine with temporary registry
    engine = CalibrationEngine(
        model_registry_path=Path("models/test_registry")
    )
    
    # Train and save first version
    logger.info("Creating version 1...")
    model_v1 = ARIMAForecaster(order=(1, 1, 1), name='ARIMA_test')
    model_v1.fit(data)
    
    version_1 = engine.save_model_version(
        model=model_v1,
        data=data,
        performance_metrics={'rmse': 5.0, 'mae': 4.0},
        hyperparameters={'order': (1, 1, 1)}
    )
    logger.info(f"Created version: {version_1.version_id}")
    
    # Activate version 1
    engine.activate_model_version(version_1.version_id)
    logger.info(f"Activated version: {version_1.version_id}")
    
    # Train and save second version
    logger.info("\nCreating version 2...")
    model_v2 = ARIMAForecaster(order=(2, 1, 1), name='ARIMA_test')
    model_v2.fit(data)
    
    version_2 = engine.save_model_version(
        model=model_v2,
        data=data,
        performance_metrics={'rmse': 4.5, 'mae': 3.8},
        hyperparameters={'order': (2, 1, 1)}
    )
    logger.info(f"Created version: {version_2.version_id}")
    
    # Activate version 2
    engine.activate_model_version(version_2.version_id)
    logger.info(f"Activated version: {version_2.version_id}")
    
    # Get active model
    active_model = engine.get_active_model('ARIMA_test')
    logger.info(f"\nActive model: {active_model.name}")
    
    # Rollback to version 1
    logger.info("\nRolling back to previous version...")
    engine.rollback_model('ARIMA_test')
    
    active_model = engine.get_active_model('ARIMA_test')
    logger.info(f"Active model after rollback: {active_model.name}")
    
    logger.info("\n✓ Model versioning test passed")
    
    # Cleanup
    import shutil
    if Path("models/test_registry").exists():
        shutil.rmtree("models/test_registry")


def test_auto_retraining():
    """Test automated retraining."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Automated Retraining")
    logger.info("=" * 60)
    
    # Generate test data
    data = generate_test_data(100)
    
    # Create calibration engine
    calibration_engine = CalibrationEngine(
        model_registry_path=Path("models/test_registry")
    )
    
    # Create retraining config
    config = RetrainingConfig(
        trigger_type='time',
        time_interval=timedelta(seconds=1),  # Very short for testing
        auto_deploy=True
    )
    
    # Create auto-retraining engine
    retraining_engine = AutoRetrainingEngine(
        calibration_engine=calibration_engine,
        config=config
    )
    
    # Create and fit initial model
    logger.info("Training initial model...")
    model = ARIMAForecaster(order=(1, 1, 1), name='ARIMA_auto')
    model.fit(data)
    
    # First retraining (should trigger due to no previous training)
    logger.info("\nAttempting first retraining...")
    success, version = retraining_engine.auto_retrain(
        model=model,
        data=data,
        hyperparameters={'order': (1, 1, 1)}
    )
    
    if success:
        logger.info(f"✓ First retraining successful: {version.version_id}")
    else:
        logger.info("First retraining skipped (expected)")
    
    # Wait a bit
    import time
    time.sleep(2)
    
    # Second retraining (should trigger due to time interval)
    logger.info("\nAttempting second retraining after time interval...")
    success, version = retraining_engine.auto_retrain(
        model=model,
        data=data,
        hyperparameters={'order': (1, 1, 1)}
    )
    
    if success:
        logger.info(f"✓ Second retraining successful: {version.version_id}")
    
    # Get retraining stats
    stats = retraining_engine.get_retraining_stats()
    logger.info(f"\nRetraining statistics:")
    logger.info(f"  Total retrainings: {stats.get('total_retrainings', 0)}")
    logger.info(f"  Successful: {stats.get('successful', 0)}")
    logger.info(f"  Failed: {stats.get('failed', 0)}")
    
    logger.info("\n✓ Automated retraining test passed")
    
    # Cleanup
    import shutil
    if Path("models/test_registry").exists():
        shutil.rmtree("models/test_registry")
    if Path("logs/retraining").exists():
        shutil.rmtree("logs/retraining")


def main():
    """Run all tests."""
    logger.info("Starting Calibration Engine Tests")
    logger.info("=" * 60)
    
    try:
        # Test 1: Backtesting
        engine, backtest_results = test_backtesting()
        
        # Test 2: Model comparison
        best_model = test_model_comparison(engine, backtest_results)
        
        # Test 3: Hyperparameter optimization
        best_params = test_hyperparameter_optimization()
        
        # Test 4: Model versioning
        test_model_versioning()
        
        # Test 5: Auto-retraining
        test_auto_retraining()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"\nTest failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
