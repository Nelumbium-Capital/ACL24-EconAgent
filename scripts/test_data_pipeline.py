"""
Test script to verify FRED data acquisition and pipeline.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fred_client import FREDClient
from src.data.pipeline import DataPipeline, MissingValueHandler, FrequencyAligner
from src.data.data_models import SeriesConfig
from src.utils.logging_config import logger


def main():
    """Test data pipeline with key economic indicators."""
    logger.info("=" * 60)
    logger.info("Testing US Risk Forecasting Data Pipeline")
    logger.info("=" * 60)
    
    # Initialize components
    fred_client = FREDClient()
    pipeline = DataPipeline(fred_client)
    
    # Add transformers
    pipeline.add_transformer(MissingValueHandler(method='ffill'))
    pipeline.add_transformer(FrequencyAligner(target_frequency='M'))
    
    # Configure series to fetch
    series_config = {
        'unemployment': SeriesConfig(
            series_id='UNRATE',
            name='Unemployment Rate',
            start_date='2020-01-01',
            end_date='2024-01-01',
            frequency='monthly'
        ),
        'inflation': SeriesConfig(
            series_id='CPIAUCSL',
            name='CPI Inflation',
            start_date='2020-01-01',
            end_date='2024-01-01',
            frequency='monthly',
            transformation='pct_change'
        ),
        'interest_rate': SeriesConfig(
            series_id='FEDFUNDS',
            name='Federal Funds Rate',
            start_date='2020-01-01',
            end_date='2024-01-01',
            frequency='monthly'
        ),
        'gdp': SeriesConfig(
            series_id='GDP',
            name='GDP',
            start_date='2020-01-01',
            end_date='2024-01-01',
            frequency='quarterly'
        )
    }
    
    # Process data
    logger.info("\nProcessing economic indicators...")
    data = pipeline.process(series_config)
    
    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"\nDataset shape: {data.shape}")
    logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
    logger.info(f"\nColumns: {list(data.columns)}")
    logger.info(f"\nFirst 5 rows:\n{data.head()}")
    logger.info(f"\nLast 5 rows:\n{data.tail()}")
    logger.info(f"\nSummary statistics:\n{data.describe()}")
    logger.info(f"\nMissing values:\n{data.isna().sum()}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Data pipeline test completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
