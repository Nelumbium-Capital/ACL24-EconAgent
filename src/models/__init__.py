"""Forecasting models."""

from src.models.base_forecaster import BaseForecaster, ForecastResult
from src.models.arima_forecaster import ARIMAForecaster, SARIMAForecaster
from src.models.ets_forecaster import (
    ETSForecaster,
    SimpleExponentialSmoothing,
    HoltLinearTrend,
    HoltWinters,
    AutoETS
)
from src.models.ensemble_forecaster import EnsembleForecaster
from src.models.calibration_engine import (
    CalibrationEngine,
    AutoRetrainingEngine,
    BacktestResult,
    ModelVersion,
    RetrainingConfig
)

# Optional deep learning models (require PyTorch)
try:
    from src.models.deep_var_forecaster import DeepVARForecaster
    from src.models.lstm_forecaster import LSTMForecaster
    HAS_TORCH = True
except ImportError:
    DeepVARForecaster = None
    LSTMForecaster = None
    HAS_TORCH = False

__all__ = [
    'BaseForecaster',
    'ForecastResult',
    'ARIMAForecaster',
    'SARIMAForecaster',
    'ETSForecaster',
    'SimpleExponentialSmoothing',
    'HoltLinearTrend',
    'HoltWinters',
    'AutoETS',
    'EnsembleForecaster',
    'CalibrationEngine',
    'AutoRetrainingEngine',
    'BacktestResult',
    'ModelVersion',
    'RetrainingConfig',
    'HAS_TORCH'
]

if HAS_TORCH:
    __all__.extend(['DeepVARForecaster', 'LSTMForecaster'])
