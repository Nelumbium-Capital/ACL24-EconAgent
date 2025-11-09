# Calibration Engine Guide

## Overview

The Calibration Engine provides comprehensive functionality for model evaluation, hyperparameter optimization, versioning, and automated retraining. It consists of two main components:

1. **CalibrationEngine**: Handles backtesting, hyperparameter optimization, model comparison, and versioning
2. **AutoRetrainingEngine**: Manages automated model retraining with configurable triggers

## Features

### CalibrationEngine

#### 1. Time-Series Cross-Validation Backtesting

Evaluate model performance using time-series cross-validation with expanding or rolling windows.

```python
from src.models import CalibrationEngine, ARIMAForecaster, LSTMForecaster

# Create models
models = [
    ARIMAForecaster(name='ARIMA', auto_order=True),
    LSTMForecaster(name='LSTM', lookback_window=12)
]

# Initialize engine
engine = CalibrationEngine(models=models)

# Run backtest
results = engine.backtest(
    data=time_series_data,
    n_splits=5,
    horizon=12,
    expanding_window=True
)

# Access results
for model_name, result in results.items():
    print(f"{model_name}:")
    print(f"  MAE: {result.average_metrics['mae']:.4f}")
    print(f"  RMSE: {result.average_metrics['rmse']:.4f}")
    print(f"  MAPE: {result.average_metrics['mape']:.2f}%")
    print(f"  Directional Accuracy: {result.average_metrics['directional_accuracy']:.2%}")
```

#### 2. Performance Metrics

The engine computes comprehensive metrics for each fold:

- **MAE** (Mean Absolute Error): Average absolute difference between predictions and actuals
- **RMSE** (Root Mean Squared Error): Square root of average squared errors
- **MAPE** (Mean Absolute Percentage Error): Average percentage error
- **Directional Accuracy**: Percentage of correct direction predictions
- **R-squared**: Coefficient of determination
- **Mean Error**: Average bias in predictions
- **Standard Error**: Standard deviation of errors

#### 3. Hyperparameter Optimization

Optimize model hyperparameters using grid search with cross-validation.

```python
# Define parameter grid
param_grid = {
    'order': [(1, 1, 1), (2, 1, 1), (1, 1, 2)],
    'seasonal': [False, True],
    'name': ['ARIMA_optimized']
}

# Optimize
best_params, best_score = engine.optimize_hyperparameters(
    model_class=ARIMAForecaster,
    data=training_data,
    param_grid=param_grid,
    n_splits=3,
    horizon=12,
    metric='rmse',
    minimize=True
)

print(f"Best parameters: {best_params}")
print(f"Best RMSE: {best_score:.4f}")
```

#### 4. Model Comparison and Selection

Compare multiple models and select the best performer.

```python
# Compare models
comparison = engine.compare_models(
    backtest_results=results,
    metric='rmse',
    minimize=True
)

print(comparison)

# Select best model
best_model_name = engine.select_best_model(
    backtest_results=results,
    metric='rmse'
)

print(f"Best model: {best_model_name}")
```

#### 5. Model Versioning and Rollback

Track model versions with performance metrics and enable rollback.

```python
# Train and save model version
model = ARIMAForecaster(order=(1, 1, 1), name='ARIMA')
model.fit(training_data)

version = engine.save_model_version(
    model=model,
    data=training_data,
    performance_metrics={'rmse': 5.2, 'mae': 4.1},
    hyperparameters={'order': (1, 1, 1)}
)

print(f"Saved version: {version.version_id}")

# Activate version for production
engine.activate_model_version(version.version_id)

# Get active model
active_model = engine.get_active_model('ARIMA')

# Rollback to previous version if needed
engine.rollback_model('ARIMA')
```

### AutoRetrainingEngine

#### 1. Configurable Triggers

Configure when models should be retrained:

```python
from src.models import AutoRetrainingEngine, RetrainingConfig
from datetime import timedelta

# Time-based trigger
config = RetrainingConfig(
    trigger_type='time',
    time_interval=timedelta(days=7),  # Retrain weekly
    auto_deploy=True
)

# Performance-based trigger
config = RetrainingConfig(
    trigger_type='performance',
    performance_metric='rmse',
    performance_degradation_pct=10.0,  # Retrain if RMSE increases by 10%
    auto_deploy=False
)

# Data-based trigger
config = RetrainingConfig(
    trigger_type='data',
    min_new_data_points=10,  # Retrain when 10+ new data points available
    auto_deploy=True
)
```

#### 2. Automated Retraining

Automatically retrain models when conditions are met:

```python
# Create retraining engine
retraining_engine = AutoRetrainingEngine(
    calibration_engine=engine,
    config=config
)

# Check if retraining is needed
should_retrain, reason = retraining_engine.should_retrain(
    model_name='ARIMA',
    current_data=latest_data,
    current_performance=current_metrics
)

if should_retrain:
    print(f"Retraining triggered: {reason}")
    
    # Perform retraining
    success, version = retraining_engine.auto_retrain(
        model=model,
        data=latest_data,
        validation_data=validation_data,
        hyperparameters={'order': (1, 1, 1)}
    )
    
    if success:
        print(f"Retraining successful: {version.version_id}")
```

#### 3. Validation Before Deployment

Models are validated before deployment to ensure quality:

```python
config = RetrainingConfig(
    trigger_type='time',
    time_interval=timedelta(days=7),
    validation_metric_threshold=5.0,  # Only deploy if RMSE < 5.0
    auto_deploy=True
)

retraining_engine = AutoRetrainingEngine(
    calibration_engine=engine,
    config=config
)

# Retraining will only deploy if validation passes
success, version = retraining_engine.auto_retrain(
    model=model,
    data=training_data,
    validation_data=validation_data
)
```

#### 4. Retraining History and Statistics

Track retraining activity over time:

```python
# Get retraining history
history = retraining_engine.get_retraining_history(
    model_name='ARIMA',
    start_date=datetime(2024, 1, 1)
)

print(history)

# Get statistics
stats = retraining_engine.get_retraining_stats()
print(f"Total retrainings: {stats['total_retrainings']}")
print(f"Successful: {stats['successful']}")
print(f"Failed: {stats['failed']}")
print(f"Deployed: {stats['deployed']}")
```

#### 5. Scheduled Retraining

Schedule periodic retraining checks:

```python
def get_latest_data():
    """Function to fetch latest training data."""
    # Fetch from database or API
    return latest_time_series_data

# Schedule retraining checks every hour
retraining_engine.schedule_retraining(
    models=[model1, model2, model3],
    data_provider=get_latest_data,
    check_interval=timedelta(hours=1)
)
```

## Complete Example

Here's a complete workflow using both engines:

```python
from src.models import (
    CalibrationEngine,
    AutoRetrainingEngine,
    RetrainingConfig,
    ARIMAForecaster,
    LSTMForecaster
)
from datetime import timedelta
import pandas as pd

# 1. Load data
data = pd.read_csv('time_series_data.csv', index_col=0, parse_dates=True)
train_data = data['2020':'2023']
validation_data = data['2024']

# 2. Create models
models = [
    ARIMAForecaster(name='ARIMA', auto_order=True),
    LSTMForecaster(name='LSTM', lookback_window=12, epochs=50)
]

# 3. Initialize calibration engine
calibration_engine = CalibrationEngine(
    models=models,
    model_registry_path=Path('models/registry')
)

# 4. Backtest models
print("Running backtest...")
backtest_results = calibration_engine.backtest(
    data=train_data,
    n_splits=5,
    horizon=12,
    expanding_window=True
)

# 5. Compare and select best model
comparison = calibration_engine.compare_models(backtest_results, metric='rmse')
print(comparison)

best_model_name = calibration_engine.select_best_model(backtest_results)
print(f"Best model: {best_model_name}")

# 6. Optimize hyperparameters for best model
if best_model_name == 'ARIMA':
    param_grid = {
        'order': [(1, 1, 1), (2, 1, 1), (1, 1, 2)],
        'auto_order': [False],
        'name': ['ARIMA_optimized']
    }
    
    best_params, best_score = calibration_engine.optimize_hyperparameters(
        model_class=ARIMAForecaster,
        data=train_data,
        param_grid=param_grid,
        n_splits=3,
        horizon=12
    )
    
    # Create optimized model
    optimized_model = ARIMAForecaster(**best_params)
else:
    optimized_model = models[1]  # Use LSTM

# 7. Train and save production model
print("Training production model...")
optimized_model.fit(train_data)

version = calibration_engine.save_model_version(
    model=optimized_model,
    data=train_data,
    performance_metrics=backtest_results[best_model_name].average_metrics,
    hyperparameters=best_params if best_model_name == 'ARIMA' else {}
)

# 8. Activate for production
calibration_engine.activate_model_version(version.version_id)
print(f"Activated version: {version.version_id}")

# 9. Set up automated retraining
retraining_config = RetrainingConfig(
    trigger_type='time',
    time_interval=timedelta(days=7),
    validation_metric_threshold=backtest_results[best_model_name].average_metrics['rmse'] * 1.1,
    auto_deploy=True
)

retraining_engine = AutoRetrainingEngine(
    calibration_engine=calibration_engine,
    config=retraining_config
)

# 10. Monitor and retrain as needed
def monitor_and_retrain():
    """Periodic monitoring and retraining."""
    # Get latest data
    latest_data = fetch_latest_data()
    
    # Check if retraining needed
    active_model = calibration_engine.get_active_model(best_model_name)
    
    success, new_version = retraining_engine.auto_retrain(
        model=active_model,
        data=latest_data,
        validation_data=validation_data
    )
    
    if success:
        print(f"Model retrained: {new_version.version_id}")
    
    # Get stats
    stats = retraining_engine.get_retraining_stats()
    print(f"Retraining stats: {stats}")

# Run monitoring
monitor_and_retrain()
```

## Best Practices

1. **Backtesting**:
   - Use expanding windows for most cases (more realistic)
   - Use rolling windows when recent data is more relevant
   - Ensure sufficient training data (at least 2x the forecast horizon)
   - Use at least 3-5 folds for reliable estimates

2. **Hyperparameter Optimization**:
   - Start with a coarse grid, then refine around best values
   - Use fewer CV folds (3) to speed up optimization
   - Consider using Bayesian optimization for large parameter spaces

3. **Model Versioning**:
   - Always save performance metrics with each version
   - Document hyperparameters for reproducibility
   - Keep at least 3-5 recent versions for rollback
   - Test rollback procedures regularly

4. **Automated Retraining**:
   - Use time-based triggers for stable environments
   - Use performance-based triggers for dynamic environments
   - Always validate before auto-deployment
   - Set up notifications for retraining events
   - Monitor retraining statistics regularly

5. **Performance Monitoring**:
   - Track multiple metrics (MAE, RMSE, directional accuracy)
   - Monitor both in-sample and out-of-sample performance
   - Set up alerts for performance degradation
   - Review retraining logs periodically

## Troubleshooting

### Issue: Backtest fails with insufficient data

**Solution**: Reduce `n_splits` or `horizon`, or provide more training data.

### Issue: Hyperparameter optimization takes too long

**Solution**: Reduce parameter grid size, use fewer CV folds, or parallelize (future enhancement).

### Issue: Model version not found

**Solution**: Check that the model was saved successfully and the registry path is correct.

### Issue: Retraining not triggering

**Solution**: Check trigger configuration and verify that conditions are met. Use `should_retrain()` to debug.

### Issue: Validation always fails

**Solution**: Adjust `validation_metric_threshold` or check validation data quality.

## API Reference

See the docstrings in `calibration_engine.py` for detailed API documentation.

## Requirements

The calibration engine requires the following dependencies:

- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- All model-specific dependencies (statsmodels, torch, etc.)

## Future Enhancements

Planned features for future versions:

1. Parallel hyperparameter optimization
2. Bayesian optimization support
3. Multi-objective optimization
4. Distributed backtesting
5. Real-time performance monitoring dashboard
6. Integration with MLflow for experiment tracking
7. A/B testing framework for model comparison
8. Automated feature selection
9. Ensemble weight optimization
10. Cloud storage integration for model registry
