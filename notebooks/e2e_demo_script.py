"""
EconAgent Risk Forecasting System - E2E Demonstration Script

This script demonstrates the complete end-to-end workflow:
1. FRED Data Acquisition → Fetch and process macroeconomic data
2. Model Training → Fit ARIMA + ETS with rolling CV ensemble
3. Forecasting → Generate 12-month forecasts with confidence intervals
4. ABM Simulation → Run LLM-based agent simulation across 4 scenarios
5. KRI Calculation → Compute Key Risk Indicators per scenario
6. Comparison & Export → Compare scenarios and export results

This can be converted to a Jupyter notebook or run as a standalone script.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

# Project imports
from src.data.fred_client import FREDClient
from src.data.pipeline import DataPipeline, MissingValueHandler, FrequencyAligner
from src.data.data_models import SeriesConfig
from src.models.arima_forecaster import ARIMAForecaster
from src.models.ets_forecaster import ETSForecaster
from src.models.ensemble_forecaster import EnsembleForecaster
from src.simulation.scenario_runner import ScenarioRunner
from src.kri.calculator import KRICalculator
from src.utils.logging_config import logger


def print_section(title: str):
    """Print formatted section header."""
    logger.info("\n" + "="*70)
    logger.info(f"  {title}")
    logger.info("="*70)


def main():
    """Run complete E2E demonstration."""
    
    print_section("EconAgent E2E Demonstration - START")
    
    # ===========================================
    # 1. FRED DATA ACQUISITION
    # ===========================================
    print_section("STEP 1: FRED Data Acquisition")
    
    fred_client = FREDClient()
    pipeline = DataPipeline(fred_client)
    
    # Add transformers
    pipeline.add_transformer(MissingValueHandler(method='ffill'))
    pipeline.add_transformer(FrequencyAligner(target_frequency='M'))
    
    # Configure series
    series_config = {
        'unemployment': SeriesConfig(
            series_id='UNRATE',
            name='Unemployment Rate',
            start_date='2018-01-01',
            end_date='2024-01-01',
            frequency='monthly'
        ),
        'inflation': SeriesConfig(
            series_id='CPIAUCSL',
            name='CPI Inflation',
            start_date='2018-01-01',
            end_date='2024-01-01',
            frequency='monthly',
            transformation='pct_change'
        ),
        'interest_rate': SeriesConfig(
            series_id='FEDFUNDS',
            name='Federal Funds Rate',
            start_date='2018-01-01',
            end_date='2024-01-01',
            frequency='monthly'
        ),
        'credit_spread': SeriesConfig(
            series_id='BAA10Y',
            name='BAA-Treasury Spread',
            start_date='2018-01-01',
            end_date='2024-01-01',
            frequency='monthly'
        )
    }
    
    # Process data
    logger.info("Fetching data from FRED...")
    economic_data = pipeline.process(series_config)
    
    logger.info(f"\n✓ Fetched {len(economic_data)} observations")
    logger.info(f"  Date range: {economic_data.index.min().date()} to {economic_data.index.max().date()}")
    logger.info(f"  Indicators: {list(economic_data.columns)}")
    
    # ===========================================
    # 2. MODEL TRAINING WITH ROLLING CV
    # ===========================================
    print_section("STEP 2: Model Training with Rolling CV Ensemble")
    
    # Select unemployment for demonstration
    target_series = economic_data['unemployment'].dropna()
    logger.info(f"Training models on {len(target_series)} observations...")
    
    # Create models
    arima = ARIMAForecaster(auto_order=True, name='ARIMA')
    ets = ETSForecaster(trend='add', seasonal=None, name='ETS')
    
    # Fit models
    logger.info("Fitting ARIMA...")
    arima.fit(target_series)
    
    logger.info("Fitting ETS...")
    ets.fit(target_series)
    
    # Create ensemble with rolling CV
    logger.info("\nCreating ensemble with rolling CV optimization...")
    ensemble = EnsembleForecaster(
        models=[arima, ets],
        weight_optimization='optimize',
        name='Ensemble'
    )
    
    ensemble.fit(target_series)
    
    logger.info(f"\n✓ Models trained successfully")
    logger.info(f"\nEnsemble weights:")
    for i, model in enumerate(ensemble.models):
        logger.info(f"  {model.name}: {ensemble.weights[i]:.4f}")
    
    # ===========================================
    # 3. GENERATE FORECASTS
    # ===========================================
    print_section("STEP 3: Generate Forecasts")
    
    forecast_horizon = 12
    logger.info(f"Generating {forecast_horizon}-month forecasts...")
    
    # Individual model forecasts
    arima_forecast = arima.forecast(horizon=forecast_horizon)
    ets_forecast = ets.forecast(horizon=forecast_horizon)
    ensemble_forecast = ensemble.forecast(horizon=forecast_horizon, return_individual=True)
    
    logger.info("✓ Forecasts generated")
    logger.info(f"\nEnsemble forecast summary:")
    logger.info(f"  Mean: {np.mean(ensemble_forecast.point_forecast):.2f}%")
    logger.info(f"  Range: {np.min(ensemble_forecast.point_forecast):.2f}% - {np.max(ensemble_forecast.point_forecast):.2f}%")
    
    # Create forecast dates
    last_date = target_series.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_horizon,
        freq='M'
    )
    
    # Save forecast
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'ensemble_forecast': ensemble_forecast.point_forecast,
        'lower_bound': ensemble_forecast.lower_bound,
        'upper_bound': ensemble_forecast.upper_bound,
        'arima_forecast': arima_forecast.point_forecast,
        'ets_forecast': ets_forecast.point_forecast
    })
    
    output_file = Path('data/processed/unemployment_forecast.csv')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Forecast saved to {output_file}")
    
    # ===========================================
    # 4. AGENT-BASED SIMULATION
    # ===========================================
    print_section("STEP 4: Agent-Based Simulation (4 Scenarios)")
    
    logger.info("Initializing scenario runner...")
    runner = ScenarioRunner(
        n_banks=10,
        n_firms=50,
        n_workers=20,  # LLM worker agents
        n_steps=100,
        use_llm_agents=False  # Set to True to enable LLM (requires NeMo/Ollama)
    )
    
    logger.info("\nRunning all scenarios (this may take a few minutes)...")
    scenario_results = runner.run_all_scenarios()
    
    logger.info("\n✓ All scenarios complete")
    
    # ===========================================
    # 5. KRI SUMMARY
    # ===========================================
    print_section("STEP 5: KRI Summary Across Scenarios")
    
    # Extract KRI summary
    kri_summary = []
    for scenario_name, kris in runner.scenario_kris.items():
        row = {
            'Scenario': scenario_name,
            'Final Default Rate': kris.get('default_rate', np.nan),
            'Final Liquidity': kris.get('system_liquidity', np.nan),
            'Avg Capital Ratio': kris.get('avg_capital_ratio', np.nan),
            'Network Stress': kris.get('network_stress', np.nan),
            'Max Default Rate': kris.get('max_default_rate', np.nan)
        }
        kri_summary.append(row)
    
    kri_df = pd.DataFrame(kri_summary)
    logger.info("\nKRI Summary:")
    logger.info(kri_df.to_string(index=False))
    
    # Save KRI summary
    kri_file = Path('data/processed/scenarios/kri_summary.csv')
    kri_df.to_csv(kri_file, index=False)
    logger.info(f"\n✓ KRI summary saved to {kri_file}")
    
    # ===========================================
    # 6. EXPORT RESULTS
    # ===========================================
    print_section("STEP 6: Export Results")
    
    runner.export_results(format='csv')
    logger.info("✓ Scenario results exported to data/processed/scenarios/")
    
    # Generate plots if matplotlib available
    try:
        for metric in ['default_rate', 'system_liquidity', 'avg_capital_ratio']:
            runner.plot_scenario_comparison(metric)
        logger.info("✓ Comparison plots generated")
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
    
    # ===========================================
    # COMPLETION
    # ===========================================
    print_section("✓ E2E DEMONSTRATION COMPLETE")
    
    logger.info("\nSummary:")
    logger.info("  ✓ Fetched 4 economic indicators from FRED (2018-2024)")
    logger.info("  ✓ Trained ARIMA + ETS with rolling CV ensemble")
    logger.info("  ✓ Generated 12-month unemployment forecasts with 95% CI")
    logger.info("  ✓ Ran ABM across 4 stress test scenarios")
    logger.info("  ✓ Computed key risk indicators per scenario")
    logger.info("  ✓ Exported all results to CSV/JSON")
    
    logger.info("\nOutput files:")
    logger.info("  - data/processed/unemployment_forecast.csv")
    logger.info("  - data/processed/scenarios/kri_summary.csv")
    logger.info("  - data/processed/scenarios/scenario_comparison.csv")
    logger.info("  - data/processed/scenarios/*_results.csv")
    
    logger.info("\nNext steps:")
    logger.info("  - Enable LLM agents: Set use_llm_agents=True")
    logger.info("  - Add more forecasting models (LSTM, Deep VAR)")
    logger.info("  - Customize scenarios with different parameters")
    logger.info("  - Integrate with real-time API for live monitoring")


if __name__ == "__main__":
    main()

