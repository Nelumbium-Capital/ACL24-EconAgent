# Classical Time-Series Forecasting Models

This module implements classical time-series forecasting models for the US Financial Risk Forecasting System.

## Overview

The module provides a unified interface for various forecasting approaches:

- **ARIMA/SARIMA**: AutoRegressive Integrated Moving Average models
- **Exponential Smoothing**: ETS models including Simple, Holt, and Holt-Winters methods
- **Base Framework**: Abstract base class with common utilities

## Architecture

All forecasters inherit from `BaseForecaster` which provides:
- Data validation and preprocessing
- Stationarity testing
- Differencing transformations
- Prediction interval calculation
- Forecast evaluation metrics

## Models

### 1. BaseForecaster (Abstract)

Base class providing common functionality for all forecasters.

**Key Methods:**
- `fit(data)`: Train the model on historical data
- `forecast(horizon, confidence_level)`: Generate forecasts
- `validate_data(data)`: Validate and clean input data
- `check_stationarity(data)`: Test for stationarity
- `evaluate_forecast(actual, predicted)`: Calculate accuracy metrics

### 2. ARIMAForecaster

ARIMA (AutoRegressive Integrated Moving Average) model with automatic order selection.

**Features:**
- Automatic order selection using AIC/BIC
- Support for non-stationary data through differencing
- Prediction intervals
- Diagnostic plots

**Usage:**
```python
from src.models import ARIMAForecaster

# Create forecaster with automatic order selection
model = ARIMAForecaster(auto_order=True)

# Fit to data
model.fit(time_series_data)

# Generate 12-step forecast
forecast = model.forecast(horizon=12, confidence_level=0.95)

print(f"Point forecast: {forecast.point_forecast}")
print(f"Lower bound: {forecast.lower_bound}")
print(f"Upper bound: {forecast.upper_bound}")
```

**Parameters:**
- `order`: ARIMA order (p, d, q) - auto-selected if None
- `auto_order`: Enable automatic order selection
- `information_criterion`: 'aic' or 'bic' for model selection

### 3. SARIMAForecaster

Seasonal ARIMA model for data with seasonal patterns.

**Features:**
- Handles seasonal patterns
- Automatic seasonal order selection
- Seasonal decomposition analysis
- Seasonal strength calculation

**Usage:**
```python
from src.models import SARIMAForecaster

# Create SARIMA forecaster for monthly data (12-month seasonality)
model = SARIMAForecaster(
    seasonal_period=12,
    auto_order=True
)

# Fit to data
model.fit(time_series_data)

# Generate forecast
forecast = model.forecast(horizon=12)
```

**Parameters:**
- `seasonal_period`: Number of periods in a season (e.g., 12 for monthly)
- `order`: Non-seasonal ARIMA order (p, d, q)
- `seasonal_order`: Seasonal order (P, D, Q, s)
- `auto_order`: Enable automatic order selection

### 4. ETSForecaster

Exponential Smoothing State Space model with configurable components.

**Features:**
- Flexible trend and seasonal components
- Additive or multiplicative components
- Damped trend option
- Box-Cox transformation support

**Usage:**
```python
from src.models import ETSForecaster

# Create ETS model with additive trend and seasonal components
model = ETSForecaster(
    trend='add',
    seasonal='add',
    seasonal_periods=12,
    damped_trend=False
)

model.fit(time_series_data)
forecast = model.forecast(horizon=12)
```

**Parameters:**
- `trend`: 'add', 'mul', or None
- `seasonal`: 'add', 'mul', or None
- `seasonal_periods`: Seasonal cycle length
- `damped_trend`: Enable damped trend

### 5. SimpleExponentialSmoothing

Simple exponential smoothing for stationary data without trend or seasonality.

**Usage:**
```python
from src.models import SimpleExponentialSmoothing

model = SimpleExponentialSmoothing()
model.fit(stationary_data)
forecast = model.forecast(horizon=12)
```

### 6. HoltLinearTrend

Holt's linear trend method for data with trend but no seasonality.

**Usage:**
```python
from src.models import HoltLinearTrend

# Standard Holt's method
model = HoltLinearTrend(damped=False)

# Or damped trend version
model_damped = HoltLinearTrend(damped=True)

model.fit(trended_data)
forecast = model.forecast(horizon=12)
```

### 7. HoltWinters

Holt-Winters method for data with both trend and seasonality.

**Usage:**
```python
from src.models import HoltWinters

# Additive seasonality
model = HoltWinters(
    seasonal_periods=12,
    trend='add',
    seasonal='add'
)

# Multiplicative seasonality
model_mult = HoltWinters(
    seasonal_periods=12,
    trend='add',
    seasonal='mul'
)

model.fit(seasonal_data)
forecast = model.forecast(horizon=12)
```

### 8. AutoETS

Automatic ETS model selection - finds the best combination of components.

**Features:**
- Automatically tests multiple configurations
- Selects best model using AIC/BIC
- Handles various data patterns

**Usage:**
```python
from src.models import AutoETS

# Automatically select best ETS configuration
model = AutoETS(
    seasonal_periods=12,
    auto_seasonal=True,
    information_criterion='aic'
)

model.fit(time_series_data)
print(f"Best configuration: {model.best_config}")

forecast = model.forecast(horizon=12)
```

## ForecastResult Object

All forecasters return a `ForecastResult` object containing:

```python
@dataclass
class ForecastResult:
    model_name: str              # Name of the forecasting model
    series_name: str             # Name of the time series
    forecast_date: datetime      # When forecast was generated
    horizon: int                 # Number of periods forecasted
    point_forecast: np.ndarray   # Point forecasts
    lower_bound: np.ndarray      # Lower confidence bound
    upper_bound: np.ndarray      # Upper confidence bound
    confidence_level: float      # Confidence level (e.g., 0.95)
    metadata: Dict[str, Any]     # Model-specific metadata
```

## Data Requirements

All models expect:
- **pandas.Series** with **DatetimeIndex**
- At least 2 observations (more recommended)
- No infinite values
- Missing values are automatically handled (forward-fill)

## Model Selection Guidelines

### Choose ARIMA when:
- Data has no clear seasonal pattern
- You need automatic model selection
- Data may be non-stationary

### Choose SARIMA when:
- Data has clear seasonal patterns
- You know the seasonal period
- You need to model both trend and seasonality

### Choose Simple Exponential Smoothing when:
- Data is stationary
- No trend or seasonality
- You need fast, simple forecasts

### Choose Holt's Linear Trend when:
- Data has a trend but no seasonality
- You want to capture trend momentum
- Consider damped version for long-term forecasts

### Choose Holt-Winters when:
- Data has both trend and seasonality
- Seasonal pattern is consistent
- You know if seasonality is additive or multiplicative

### Choose AutoETS when:
- You're unsure about data patterns
- You want automatic model selection
- You need a robust baseline

## Evaluation Metrics

Use `BaseForecaster.evaluate_forecast()` to calculate:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Percentage of correct direction predictions

```python
# Evaluate forecast accuracy
metrics = model.evaluate_forecast(actual_values, predicted_values)
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAE: {metrics['mae']:.2f}")
print(f"MAPE: {metrics['mape']:.2f}%")
print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
```

## Example: Complete Workflow

```python
import pandas as pd
from src.models import ARIMAForecaster, SARIMAForecaster, AutoETS

# Load your time series data
data = pd.read_csv('data.csv', index_col='date', parse_dates=True)['value']

# Try multiple models
models = {
    'ARIMA': ARIMAForecaster(auto_order=True),
    'SARIMA': SARIMAForecaster(seasonal_period=12, auto_order=True),
    'AutoETS': AutoETS(seasonal_periods=12)
}

# Fit all models
for name, model in models.items():
    print(f"Fitting {name}...")
    model.fit(data)
    print(f"  AIC: {model.fitted_model.aic:.2f}")

# Generate forecasts
forecasts = {}
for name, model in models.items():
    forecasts[name] = model.forecast(horizon=12)

# Compare forecasts
for name, forecast in forecasts.items():
    print(f"\n{name} Forecast:")
    print(f"  Mean: {forecast.point_forecast.mean():.2f}")
    print(f"  Std: {forecast.point_forecast.std():.2f}")
```

## Testing

Run validation tests:
```bash
python scripts/validate_classical_models.py
```

Run full integration tests (requires dependencies):
```bash
python scripts/test_classical_models.py
```

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- statsmodels >= 0.14.0
- pmdarima >= 2.0.0
- scipy >= 1.10.0

## References

- Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). Time Series Analysis: Forecasting and Control
- Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice
- statsmodels documentation: https://www.statsmodels.org/
- pmdarima documentation: https://alkaline-ml.com/pmdarima/

## Future Enhancements

Planned additions:
- Prophet model integration
- Neural network forecasters (LSTM, Transformer)
- Ensemble methods
- Online learning capabilities
- GPU acceleration for large-scale forecasting
