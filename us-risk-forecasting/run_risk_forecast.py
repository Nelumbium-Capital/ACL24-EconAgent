"""
Main script to run US Financial Risk Forecasting System MVP.
Demonstrates end-to-end workflow: Data -> Forecasts -> KRIs -> Risk Assessment
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data.fred_client import FREDClient
from src.data.pipeline import DataPipeline, MissingValueHandler, FrequencyAligner
from src.data.data_models import SeriesConfig
from src.kri.calculator import KRICalculator
from src.kri.definitions import kri_registry
from src.models.llm_forecaster import LLMEnsembleForecaster
from src.utils.logging_config import logger


def print_section(title: str):
    """Print formatted section header."""
    logger.info("\n" + "=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def main():
    """Run complete risk forecasting workflow."""
    print_section("US FINANCIAL RISK FORECASTING SYSTEM - MVP")
    
    # ========================================================================
    # STEP 1: DATA ACQUISITION
    # ========================================================================
    print_section("STEP 1: Data Acquisition from FRED")
    
    fred_client = FREDClient()
    pipeline = DataPipeline(fred_client)
    
    # Add transformers
    pipeline.add_transformer(MissingValueHandler(method='ffill'))
    pipeline.add_transformer(FrequencyAligner(target_frequency='M'))
    
    # Configure key economic indicators
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
    economic_data = pipeline.process(series_config)
    
    logger.info(f"\n✓ Fetched {len(economic_data)} observations")
    logger.info(f"  Date range: {economic_data.index.min().date()} to {economic_data.index.max().date()}")
    logger.info(f"  Indicators: {list(economic_data.columns)}")
    logger.info(f"\nLatest values:")
    for col in economic_data.columns:
        logger.info(f"  {col}: {economic_data[col].iloc[-1]:.4f}")
    
    # ========================================================================
    # STEP 2: GENERATE FORECASTS
    # ========================================================================
    print_section("STEP 2: Generate Economic Forecasts")
    
    # Simple forecasts for each indicator
    forecast_horizon = 6
    forecasts_dict = {}
    
    logger.info(f"\nGenerating {forecast_horizon}-month forecasts...")
    
    # Use LLM ensemble forecaster
    llm_forecaster = LLMEnsembleForecaster()
    
    for col in economic_data.columns:
        series = economic_data[col].dropna().values
        
        logger.info(f"\nForecasting {col}...")
        
        try:
            # Try LLM forecast
            result = llm_forecaster.forecast(
                series=series,
                horizon=forecast_horizon,
                series_name=col,
                use_llm=True
            )
            forecasts_dict[col] = result['ensemble']
            
            if 'llm_reasoning' in result:
                logger.info(f"  LLM reasoning: {result['llm_reasoning'][:100]}...")
        
        except Exception as e:
            logger.warning(f"  LLM forecast failed, using simple trend: {e}")
            # Fallback: simple trend
            if len(series) >= 2:
                trend = series[-1] - series[-2]
                forecasts_dict[col] = np.array([series[-1] + trend * (i+1) for i in range(forecast_horizon)])
            else:
                forecasts_dict[col] = np.full(forecast_horizon, series[-1])
    
    # Create forecast DataFrame
    forecast_dates = pd.date_range(
        start=economic_data.index[-1] + pd.DateOffset(months=1),
        periods=forecast_horizon,
        freq='M'
    )
    forecasts_df = pd.DataFrame(forecasts_dict, index=forecast_dates)
    
    logger.info(f"\n✓ Generated forecasts for {len(forecasts_df.columns)} indicators")
    logger.info(f"\nForecast summary:")
    logger.info(forecasts_df.to_string())
    
    # ========================================================================
    # STEP 3: COMPUTE KEY RISK INDICATORS (KRIs)
    # ========================================================================
    print_section("STEP 3: Compute Key Risk Indicators")
    
    kri_calc = KRICalculator()
    
    # Combine historical and forecast data for KRI calculation
    combined_data = pd.concat([economic_data.tail(12), forecasts_df])
    
    # Compute all KRIs
    kris = kri_calc.compute_all_kris(
        forecasts=combined_data,
        simulation_results=None,  # Would come from Mesa simulation
        portfolio_data=None,
        balance_sheet=None
    )
    
    logger.info(f"\n✓ Computed {len(kris)} KRIs")
    logger.info("\nKRI Values:")
    for kri_name, value in kris.items():
        kri_def = kri_registry.get_kri(kri_name)
        unit = kri_def.unit if kri_def else ""
        logger.info(f"  {kri_name}: {value:.2f} {unit}")
    
    # ========================================================================
    # STEP 4: RISK ASSESSMENT
    # ========================================================================
    print_section("STEP 4: Risk Assessment & Threshold Evaluation")
    
    # Evaluate against thresholds
    risk_levels = kri_calc.evaluate_thresholds(kris)
    
    logger.info("\nRisk Level Assessment:")
    for kri_name, risk_level in risk_levels.items():
        kri_def = kri_registry.get_kri(kri_name)
        value = kris[kri_name]
        unit = kri_def.unit if kri_def else ""
        
        # Color code by risk level
        level_str = risk_level.value.upper()
        logger.info(f"  [{level_str:8}] {kri_name}: {value:.2f} {unit}")
    
    # Summary statistics
    risk_counts = {}
    for level in risk_levels.values():
        risk_counts[level.value] = risk_counts.get(level.value, 0) + 1
    
    logger.info("\nRisk Summary:")
    for level, count in sorted(risk_counts.items()):
        logger.info(f"  {level.upper()}: {count} KRIs")
    
    # ========================================================================
    # STEP 5: GENERATE RISK REPORT
    # ========================================================================
    print_section("STEP 5: Risk Report Summary")
    
    # Identify critical risks
    critical_kris = [name for name, level in risk_levels.items() 
                     if level.value in ['critical', 'high']]
    
    if critical_kris:
        logger.info("\n⚠️  ATTENTION REQUIRED:")
        for kri_name in critical_kris:
            value = kris[kri_name]
            level = risk_levels[kri_name].value
            kri_def = kri_registry.get_kri(kri_name)
            
            logger.info(f"\n  • {kri_name.upper()}")
            logger.info(f"    Current Value: {value:.2f} {kri_def.unit}")
            logger.info(f"    Risk Level: {level.upper()}")
            logger.info(f"    Category: {kri_def.category.value}")
            logger.info(f"    Description: {kri_def.description}")
    else:
        logger.info("\n✓ All risk indicators within acceptable ranges")
    
    # Key insights
    logger.info("\nKey Insights:")
    logger.info(f"  • Latest unemployment: {economic_data['unemployment'].iloc[-1]:.1f}%")
    logger.info(f"  • Forecast unemployment (6mo): {forecasts_df['unemployment'].iloc[-1]:.1f}%")
    logger.info(f"  • Interest rate: {economic_data['interest_rate'].iloc[-1]:.2f}%")
    logger.info(f"  • Credit spread: {economic_data['credit_spread'].iloc[-1]:.2f}%")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print_section("✓ RISK FORECASTING COMPLETE")
    
    logger.info("\nSystem Components Demonstrated:")
    logger.info("  ✓ FRED data acquisition with caching")
    logger.info("  ✓ Data pipeline with ETL transformations")
    logger.info("  ✓ LLM-based forecasting with Nemotron")
    logger.info("  ✓ KRI calculation across risk categories")
    logger.info("  ✓ Risk threshold evaluation")
    logger.info("  ✓ Automated risk reporting")
    
    logger.info("\nData saved to:")
    logger.info(f"  • Cache: data/cache/")
    logger.info(f"  • Processed: data/processed/")
    logger.info(f"  • Logs: logs/risk_forecasting.log")
    
    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
