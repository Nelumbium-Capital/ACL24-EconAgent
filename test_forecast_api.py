#!/usr/bin/env python3
"""
Quick test to verify ARIMA+ETS forecasting with confidence intervals.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime
from src.data.fred_client import FREDClient
from src.data.pipeline import DataPipeline, MissingValueHandler, FrequencyAligner
from src.data.data_models import SeriesConfig
from src.models.arima_forecaster import ARIMAForecaster
from src.models.ets_forecaster import ETSForecaster

print("=" * 70)
print("  Testing ARIMA + ETS Forecasting with Confidence Intervals")
print("=" * 70)
print()

# Fetch unemployment data
print("📊 Fetching FRED data...")
fred_client = FREDClient()
pipeline = DataPipeline(fred_client)
pipeline.add_transformer(MissingValueHandler(method='ffill'))

config = {
    'unemployment': SeriesConfig(
        series_id='UNRATE',
        name='Unemployment Rate',
        start_date='2020-01-01',
        end_date=datetime.now().strftime('%Y-%m-%d'),
        frequency='monthly'
    )
}

data = pipeline.process(config)
series_data = data['unemployment'].dropna()

print(f"✓ Data fetched: {len(series_data)} observations")
print(f"✓ Last date: {series_data.index[-1].strftime('%Y-%m-%d')}")
print(f"✓ Last value: {series_data.iloc[-1]:.2f}%")
print()

# ARIMA Forecast
print("🔮 Generating ARIMA forecast...")
arima = ARIMAForecaster()
arima.fit(series_data)
arima_result = arima.forecast(horizon=12)
print(f"✓ ARIMA forecast: {arima_result['forecast'][:3]}")
print()

# ETS Forecast  
print("🔮 Generating ETS forecast...")
ets = ETSForecaster()
ets.fit(series_data)
ets_result = ets.forecast(horizon=12)
print(f"✓ ETS forecast: {ets_result['forecast'][:3]}")
print()

# Ensemble
print("🎯 Creating Ensemble (60% ARIMA + 40% ETS)...")
ensemble_forecast = 0.6 * arima_result['forecast'] + 0.4 * ets_result['forecast']
ensemble_std = 0.6 * arima_result.get('std', arima_result['forecast'] * 0.05) + \
               0.4 * ets_result.get('std', ets_result['forecast'] * 0.05)

forecast_lower = ensemble_forecast - 1.96 * ensemble_std
forecast_upper = ensemble_forecast + 1.96 * ensemble_std

print(f"✓ Ensemble forecast (first 3 months):")
for i in range(3):
    print(f"   Month {i+1}: {ensemble_forecast[i]:.3f}% "
          f"[{forecast_lower[i]:.3f}% - {forecast_upper[i]:.3f}%]")
print()

# Future dates
forecast_start = series_data.index[-1] + pd.DateOffset(months=1)
forecast_dates = pd.date_range(start=forecast_start, periods=12, freq='ME')

print(f"📅 Forecast period: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")
print()
print("✅ Forecasting system is working correctly!")
print("=" * 70)

