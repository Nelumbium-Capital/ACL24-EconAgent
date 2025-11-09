"""
Test that forecasts are actually generating real predictions, not just repeating last values.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
os.environ['FRED_API_KEY'] = 'bcc1a43947af1745a35bfb3b7132b7c6'

import numpy as np
import pandas as pd
from src.data.fred_client import FREDClient
from src.data.pipeline import DataPipeline, MissingValueHandler, FrequencyAligner
from src.data.data_models import SeriesConfig
from src.models.arima_forecaster import ARIMAForecaster
from src.models.ets_forecaster import ETSForecaster
from src.models.llm_forecaster import LLMEnsembleForecaster

def test_forecasts():
    print("=" * 70)
    print("TESTING REAL FORECAST PREDICTIONS")
    print("=" * 70)
    
    # Fetch real data
    print("\n1. Fetching unemployment data...")
    fred_client = FREDClient()
    pipeline = DataPipeline(fred_client)
    pipeline.add_transformer(MissingValueHandler(method='ffill'))
    pipeline.add_transformer(FrequencyAligner(target_frequency='M'))
    
    series_config = {
        'unemployment': SeriesConfig(
            series_id='UNRATE',
            name='Unemployment Rate',
            start_date='2020-01-01',
            end_date='2024-01-01',
            frequency='monthly'
        )
    }
    
    data = pipeline.process(series_config)
    series = data['unemployment'].dropna()
    
    print(f"✓ Fetched {len(series)} observations")
    print(f"  Last 5 values: {series.tail().values}")
    print(f"  Last value: {series.iloc[-1]:.2f}%")
    
    # Test ARIMA
    print("\n2. Testing ARIMA forecaster...")
    arima = ARIMAForecaster(auto_order=False, order=(2, 1, 2))
    arima.fit(series)
    arima_result = arima.forecast(horizon=12)
    arima_forecast = arima_result.point_forecast
    
    print(f"✓ ARIMA forecast generated")
    print(f"  First 3 forecasts: {arima_forecast[:3]}")
    print(f"  Last 3 forecasts: {arima_forecast[-3:]}")
    print(f"  Forecast range: {arima_forecast.min():.2f}% to {arima_forecast.max():.2f}%")
    
    # Check if it's not just repeating last value
    last_value = series.iloc[-1]
    is_varying = not np.allclose(arima_forecast, last_value, atol=0.01)
    if is_varying:
        print(f"  ✓ ARIMA is generating varying predictions (not just repeating {last_value:.2f}%)")
    else:
        print(f"  ✗ WARNING: ARIMA is just repeating last value {last_value:.2f}%")
    
    # Test ETS
    print("\n3. Testing ETS forecaster...")
    ets = ETSForecaster(trend='add', seasonal=None)
    ets.fit(series)
    ets_result = ets.forecast(horizon=12)
    ets_forecast = ets_result.point_forecast
    
    print(f"✓ ETS forecast generated")
    print(f"  First 3 forecasts: {ets_forecast[:3]}")
    print(f"  Last 3 forecasts: {ets_forecast[-3:]}")
    print(f"  Forecast range: {ets_forecast.min():.2f}% to {ets_forecast.max():.2f}%")
    
    is_varying = not np.allclose(ets_forecast, last_value, atol=0.01)
    if is_varying:
        print(f"  ✓ ETS is generating varying predictions")
    else:
        print(f"  ✗ WARNING: ETS is just repeating last value")
    
    # Test LLM Ensemble
    print("\n4. Testing LLM Ensemble forecaster...")
    try:
        llm = LLMEnsembleForecaster()
        llm_result = llm.forecast(
            series=series.values,
            horizon=12,
            series_name='unemployment',
            use_llm=False
        )
        llm_forecast = llm_result['ensemble']
        
        print(f"✓ LLM Ensemble forecast generated")
        print(f"  First 3 forecasts: {llm_forecast[:3]}")
        print(f"  Last 3 forecasts: {llm_forecast[-3:]}")
        print(f"  Forecast range: {llm_forecast.min():.2f}% to {llm_forecast.max():.2f}%")
        
        is_varying = not np.allclose(llm_forecast, last_value, atol=0.01)
        if is_varying:
            print(f"  ✓ LLM Ensemble is generating varying predictions")
        else:
            print(f"  ✗ WARNING: LLM Ensemble is just repeating last value")
    except Exception as e:
        print(f"  ✗ LLM Ensemble failed: {e}")
    
    # Compare models
    print("\n5. Comparing model predictions...")
    ensemble = (arima_forecast + ets_forecast) / 2
    
    print(f"  ARIMA mean: {arima_forecast.mean():.2f}%")
    print(f"  ETS mean: {ets_forecast.mean():.2f}%")
    print(f"  Ensemble mean: {ensemble.mean():.2f}%")
    print(f"  Historical mean (last 12): {series.tail(12).mean():.2f}%")
    print(f"  Last value: {last_value:.2f}%")
    
    # Check for diversity
    arima_std = arima_forecast.std()
    ets_std = ets_forecast.std()
    
    print(f"\n6. Forecast variability:")
    print(f"  ARIMA std dev: {arima_std:.3f}%")
    print(f"  ETS std dev: {ets_std:.3f}%")
    
    if arima_std > 0.01 or ets_std > 0.01:
        print(f"  ✓ Forecasts show meaningful variation")
    else:
        print(f"  ✗ WARNING: Forecasts show very little variation")
    
    print("\n" + "=" * 70)
    print("FORECAST TEST COMPLETE")
    print("=" * 70)
    
    return True

if __name__ == '__main__':
    test_forecasts()
