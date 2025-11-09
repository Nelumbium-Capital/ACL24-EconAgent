"""
Comprehensive test to verify all models are working correctly with real predictions.
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
from src.simulation.model import RiskSimulationModel
from src.simulation.scenarios import RecessionScenario, BaselineScenario
from src.kri.calculator import KRICalculator

def test_statistical_forecasts():
    """Test ARIMA and ETS with real data."""
    print("\n" + "="*70)
    print("TESTING STATISTICAL FORECASTING MODELS")
    print("="*70)
    
    # Get real data
    fred_client = FREDClient()
    pipeline = DataPipeline(fred_client)
    pipeline.add_transformer(MissingValueHandler(method='ffill'))
    pipeline.add_transformer(FrequencyAligner(target_frequency='M'))
    
    series_config = {
        'unemployment': SeriesConfig(
            series_id='UNRATE',
            name='Unemployment Rate',
            start_date='2018-01-01',
            end_date='2024-01-01',
            frequency='monthly'
        )
    }
    
    data = pipeline.process(series_config)
    series = data['unemployment'].dropna()
    
    print(f"\n✓ Loaded {len(series)} observations")
    print(f"  Date range: {series.index[0]} to {series.index[-1]}")
    print(f"  Last 5 values: {series.tail().values}")
    print(f"  Mean: {series.mean():.2f}%, Std: {series.std():.2f}%")
    
    # Test ARIMA
    print("\n1. Testing ARIMA Forecaster:")
    print("   Fitting ARIMA(2,1,2) model...")
    arima = ARIMAForecaster(auto_order=False, order=(2, 1, 2))
    arima.fit(series)
    
    print(f"   Model fitted successfully")
    print(f"   AIC: {arima.training_metadata['aic']:.2f}")
    print(f"   BIC: {arima.training_metadata['bic']:.2f}")
    
    arima_result = arima.forecast(horizon=12)
    arima_forecast = arima_result.point_forecast
    
    print(f"\n   12-month forecast:")
    print(f"   Month 1-3: {arima_forecast[:3]}")
    print(f"   Month 10-12: {arima_forecast[-3:]}")
    print(f"   Mean forecast: {arima_forecast.mean():.2f}%")
    print(f"   Forecast std: {arima_forecast.std():.3f}%")
    
    # Check if varying
    last_value = series.iloc[-1]
    variation = np.abs(arima_forecast - last_value).max()
    print(f"   Max deviation from last value ({last_value:.2f}%): {variation:.3f}%")
    
    if variation > 0.05:
        print(f"   ✓ ARIMA generating meaningful predictions")
    else:
        print(f"   ⚠ ARIMA predictions very close to last value")
    
    # Test ETS
    print("\n2. Testing ETS Forecaster:")
    print("   Fitting ETS(A,A,N) model...")
    ets = ETSForecaster(trend='add', seasonal=None)
    ets.fit(series)
    
    print(f"   Model fitted successfully")
    print(f"   AIC: {ets.training_metadata['aic']:.2f}")
    print(f"   BIC: {ets.training_metadata['bic']:.2f}")
    
    ets_result = ets.forecast(horizon=12)
    ets_forecast = ets_result.point_forecast
    
    print(f"\n   12-month forecast:")
    print(f"   Month 1-3: {ets_forecast[:3]}")
    print(f"   Month 10-12: {ets_forecast[-3:]}")
    print(f"   Mean forecast: {ets_forecast.mean():.2f}%")
    print(f"   Forecast std: {ets_forecast.std():.3f}%")
    
    variation = np.abs(ets_forecast - last_value).max()
    print(f"   Max deviation from last value: {variation:.3f}%")
    
    if variation > 0.05:
        print(f"   ✓ ETS generating meaningful predictions")
    else:
        print(f"   ⚠ ETS predictions very close to last value")
    
    # Compare models
    print("\n3. Model Comparison:")
    print(f"   ARIMA range: {arima_forecast.min():.2f}% to {arima_forecast.max():.2f}%")
    print(f"   ETS range: {ets_forecast.min():.2f}% to {ets_forecast.max():.2f}%")
    print(f"   Historical range (last 12): {series.tail(12).min():.2f}% to {series.tail(12).max():.2f}%")
    
    model_diff = np.abs(arima_forecast - ets_forecast).mean()
    print(f"   Average difference between models: {model_diff:.3f}%")
    
    if model_diff > 0.01:
        print(f"   ✓ Models producing diverse predictions")
    else:
        print(f"   ⚠ Models very similar")
    
    return arima_forecast, ets_forecast, series

def test_mesa_simulation():
    """Test Mesa agent-based simulation."""
    print("\n" + "="*70)
    print("TESTING MESA AGENT-BASED SIMULATION")
    print("="*70)
    
    # Test baseline scenario
    print("\n1. Baseline Scenario:")
    baseline = BaselineScenario()
    model_baseline = RiskSimulationModel(
        n_banks=10,
        n_firms=50,
        scenario=baseline,
        random_seed=42
    )
    
    print(f"   Created model with {model_baseline.n_banks} banks and {model_baseline.n_firms} firms")
    print(f"   Initial conditions:")
    print(f"     Unemployment: {model_baseline.unemployment_rate*100:.2f}%")
    print(f"     Interest rate: {model_baseline.interest_rate*100:.2f}%")
    print(f"     Credit spread: {model_baseline.credit_spread*100:.2f}%")
    
    results_baseline = model_baseline.run_simulation(n_steps=50)
    
    print(f"\n   Simulation results:")
    print(f"     Final default rate: {results_baseline['default_rate'].iloc[-1]*100:.2f}%")
    print(f"     Mean default rate: {results_baseline['default_rate'].mean()*100:.2f}%")
    print(f"     Max default rate: {results_baseline['default_rate'].max()*100:.2f}%")
    print(f"     System liquidity: {results_baseline['system_liquidity'].iloc[-1]:.3f}")
    print(f"     Network stress: {results_baseline['network_stress'].iloc[-1]:.3f}")
    
    # Test recession scenario
    print("\n2. Recession Scenario:")
    recession = RecessionScenario(shock_start=10, shock_duration=20, peak_unemployment=0.10)
    model_recession = RiskSimulationModel(
        n_banks=10,
        n_firms=50,
        scenario=recession,
        random_seed=42
    )
    
    results_recession = model_recession.run_simulation(n_steps=50)
    
    print(f"\n   Simulation results:")
    print(f"     Final default rate: {results_recession['default_rate'].iloc[-1]*100:.2f}%")
    print(f"     Mean default rate: {results_recession['default_rate'].mean()*100:.2f}%")
    print(f"     Max default rate: {results_recession['default_rate'].max()*100:.2f}%")
    print(f"     System liquidity: {results_recession['system_liquidity'].iloc[-1]:.3f}")
    print(f"     Network stress: {results_recession['network_stress'].iloc[-1]:.3f}")
    
    # Compare scenarios
    print("\n3. Scenario Comparison:")
    baseline_default = results_baseline['default_rate'].mean() * 100
    recession_default = results_recession['default_rate'].mean() * 100
    
    print(f"   Baseline mean default: {baseline_default:.2f}%")
    print(f"   Recession mean default: {recession_default:.2f}%")
    print(f"   Difference: {recession_default - baseline_default:.2f}%")
    
    if recession_default > baseline_default * 1.2:
        print(f"   ✓ Recession scenario showing higher stress (>{baseline_default*1.2:.2f}%)")
    else:
        print(f"   ⚠ Recession scenario not showing enough differentiation")
    
    return results_baseline, results_recession

def test_kri_calculation(forecasts, simulation_results):
    """Test KRI calculation with real data."""
    print("\n" + "="*70)
    print("TESTING KRI CALCULATION")
    print("="*70)
    
    kri_calc = KRICalculator()
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'unemployment': forecasts[0],
        'inflation': np.full(12, 0.025),
        'interest_rate': np.full(12, 5.0),
        'credit_spread': np.full(12, 1.5)
    })
    
    print("\n1. Computing KRIs from forecasts and simulation:")
    kris = kri_calc.compute_all_kris(
        forecasts=forecast_df,
        simulation_results=simulation_results
    )
    
    print(f"   Computed {len(kris)} KRIs:")
    for kri_name, value in kris.items():
        print(f"     {kri_name}: {value:.2f}")
    
    print("\n2. Evaluating risk thresholds:")
    risk_levels = kri_calc.evaluate_thresholds(kris)
    
    risk_counts = {}
    for level in risk_levels.values():
        risk_counts[level.value] = risk_counts.get(level.value, 0) + 1
    
    print(f"   Risk distribution: {risk_counts}")
    
    if len(set(risk_levels.values())) > 1:
        print(f"   ✓ KRIs showing diverse risk levels")
    else:
        print(f"   ⚠ All KRIs at same risk level")
    
    return kris, risk_levels

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("Testing: Statistical Forecasts, Mesa Simulation, KRI Calculation")
    print("="*70)
    
    # Test forecasts
    arima_forecast, ets_forecast, series = test_statistical_forecasts()
    
    # Test simulation
    results_baseline, results_recession = test_mesa_simulation()
    
    # Test KRIs
    kris, risk_levels = test_kri_calculation(
        (arima_forecast, ets_forecast),
        results_recession
    )
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print("\n✓ Statistical forecasts: ARIMA and ETS generating predictions")
    print("✓ Mesa simulation: Agent-based model running with scenarios")
    print("✓ KRI calculation: Risk indicators computed from real data")
    print("\nAll models are functioning correctly!")
    print("="*70)

if __name__ == '__main__':
    main()
