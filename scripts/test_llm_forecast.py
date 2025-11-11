"""
Test LLM-based time series forecasting with Nemotron.
"""
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.llm_forecaster import NemotronTimeSeriesForecaster, LLMEnsembleForecaster
from src.data.fred_client import FREDClient
from src.utils.logging_config import logger


def main():
    """Test LLM forecasting on real economic data."""
    logger.info("=" * 60)
    logger.info("Testing LLM Time Series Forecasting with Nemotron")
    logger.info("=" * 60)
    
    # Fetch real unemployment data
    logger.info("\nFetching unemployment data from FRED...")
    fred_client = FREDClient()
    unemployment_data = fred_client.fetch_series(
        'UNRATE',
        '2020-01-01',
        '2024-01-01'
    )
    
    # Prepare data
    series = unemployment_data['value'].values
    logger.info(f"Historical data: {len(series)} observations")
    logger.info(f"Recent values: {series[-5:]}")
    
    # Test 1: Single LLM forecast
    logger.info("\n" + "=" * 60)
    logger.info("Test 1: Single LLM Forecast")
    logger.info("=" * 60)
    
    forecaster = NemotronTimeSeriesForecaster()
    
    try:
        forecasts, reasoning = forecaster.forecast(
            series=series,
            horizon=6,
            series_name="US Unemployment Rate",
            context_info="Monthly unemployment rate as percentage of labor force"
        )
        
        logger.info(f"\nForecasts (next 6 months): {forecasts}")
        logger.info(f"Reasoning: {reasoning}")
    
    except Exception as e:
        logger.error(f"LLM forecast failed: {e}")
        logger.info("This is expected if Nemotron/Ollama is not running")
    
    # Test 2: Ensemble forecast
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Ensemble Forecast (LLM + Traditional)")
    logger.info("=" * 60)
    
    ensemble = LLMEnsembleForecaster()
    
    try:
        results = ensemble.forecast(
            series=series,
            horizon=6,
            series_name="US Unemployment Rate",
            use_llm=True
        )
        
        logger.info(f"\nNaive forecast: {results['forecasts']['naive']}")
        logger.info(f"Trend forecast: {results['forecasts'].get('trend', 'N/A')}")
        if 'llm' in results['forecasts']:
            logger.info(f"LLM forecast: {results['forecasts']['llm']}")
            logger.info(f"LLM reasoning: {results.get('llm_reasoning', 'N/A')}")
        logger.info(f"Ensemble forecast: {results['ensemble']}")
    
    except Exception as e:
        logger.error(f"Ensemble forecast failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ LLM forecasting test completed!")
    logger.info("=" * 60)
    logger.info("\nNote: For full LLM functionality, ensure Nemotron or Ollama is running:")
    logger.info("  - Nemotron: http://localhost:8000/v1")
    logger.info("  - Ollama: http://localhost:11434/v1")


if __name__ == "__main__":
    main()
