"""
Data pipeline for ETL operations on time-series data.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from src.data.fred_client import FREDClient
from src.data.data_models import SeriesConfig, DatasetMetadata
from src.utils.logging_config import logger
from config import settings


class DataTransformer:
    """Base class for data transformations."""
    
    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply transformation to data."""
        raise NotImplementedError


class MissingValueHandler(DataTransformer):
    """Handle missing values in time series."""
    
    def __init__(self, method: str = 'ffill'):
        """
        Initialize handler.
        
        Args:
            method: Method for handling missing values ('ffill', 'interpolate', 'drop')
        """
        self.method = method
        logger.info(f"MissingValueHandler initialized with method: {method}")
    
    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply missing value handling."""
        result = {}
        
        for name, df in data.items():
            if df.empty:
                result[name] = df
                continue
            
            missing_before = df['value'].isna().sum()
            
            if self.method == 'ffill':
                df['value'] = df['value'].fillna(method='ffill')
            elif self.method == 'interpolate':
                df['value'] = df['value'].interpolate(method='linear')
            elif self.method == 'drop':
                df = df.dropna()
            
            missing_after = df['value'].isna().sum()
            
            if missing_before > 0:
                logger.info(
                    f"{name}: Handled {missing_before} missing values "
                    f"({missing_after} remaining)"
                )
            
            result[name] = df
        
        return result


class FrequencyAligner(DataTransformer):
    """Align series to common frequency."""
    
    def __init__(self, target_frequency: str = 'M'):
        """
        Initialize aligner.
        
        Args:
            target_frequency: Target frequency ('D', 'M', 'Q', 'Y')
        """
        self.target_frequency = target_frequency
        logger.info(f"FrequencyAligner initialized with frequency: {target_frequency}")
    
    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Resample all series to target frequency."""
        result = {}
        
        for name, df in data.items():
            if df.empty:
                result[name] = df
                continue
            
            original_freq = pd.infer_freq(df.index)
            
            # Ensure value column is numeric and drop non-numeric
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            
            # Select only numeric columns for resampling
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_numeric = df[numeric_cols]
            
            # Resample to target frequency
            df_resampled = df_numeric.resample(self.target_frequency).mean()
            
            logger.debug(
                f"{name}: Resampled from {original_freq} to {self.target_frequency} "
                f"({len(df)} -> {len(df_resampled)} observations)"
            )
            
            result[name] = df_resampled
        
        return result


class DataPipeline:
    """
    ETL pipeline for cleaning, aligning, and versioning time-series data.
    """
    
    def __init__(
        self, 
        fred_client: Optional[FREDClient] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            fred_client: FRED client for data fetching
            output_dir: Directory for versioned datasets
        """
        self.fred_client = fred_client or FREDClient()
        self.output_dir = output_dir or Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.transformers: List[DataTransformer] = []
        
        logger.info(f"DataPipeline initialized with output dir: {self.output_dir}")
    
    def add_transformer(self, transformer: DataTransformer):
        """Add a data transformation step to the pipeline."""
        self.transformers.append(transformer)
        logger.debug(f"Added transformer: {transformer.__class__.__name__}")
    
    def process(
        self, 
        series_config: Dict[str, SeriesConfig],
        save_version: bool = True
    ) -> pd.DataFrame:
        """
        Process multiple series through the pipeline.
        
        Args:
            series_config: Dictionary mapping series names to fetch configurations
            save_version: Whether to save versioned dataset
            
        Returns:
            Aligned DataFrame with all series at consistent frequency
        """
        logger.info(f"Processing {len(series_config)} series through pipeline")
        
        # Step 1: Fetch all series
        raw_data = self._fetch_series(series_config)
        
        # Step 2: Apply transformations
        processed_data = raw_data
        for transformer in self.transformers:
            logger.debug(f"Applying transformer: {transformer.__class__.__name__}")
            processed_data = transformer.transform(processed_data)
        
        # Step 3: Align to common date range and merge
        aligned_data = self._align_and_merge(processed_data)
        
        # Step 4: Version the dataset
        if save_version and not aligned_data.empty:
            metadata = self._create_metadata(series_config, aligned_data)
            self._save_versioned_dataset(aligned_data, metadata)
        
        logger.info(
            f"Pipeline complete: {len(aligned_data)} observations, "
            f"{len(aligned_data.columns)} series"
        )
        
        return aligned_data
    
    def _fetch_series(
        self, 
        series_config: Dict[str, SeriesConfig]
    ) -> Dict[str, pd.DataFrame]:
        """Fetch all configured series."""
        logger.info("Fetching series from FRED")
        
        # Prepare fetch parameters
        series_ids = [config.series_id for config in series_config.values()]
        
        # Use earliest start date and latest end date
        start_dates = [config.start_date for config in series_config.values()]
        end_dates = [config.end_date for config in series_config.values()]
        start_date = min(start_dates)
        end_date = max(end_dates)
        
        # Fetch data
        raw_data = self.fred_client.fetch_multiple_series(
            series_ids, 
            start_date, 
            end_date
        )
        
        # Map back to configured names and apply transformations
        result = {}
        for name, config in series_config.items():
            df = raw_data.get(config.series_id, pd.DataFrame())
            
            if not df.empty and config.transformation:
                df = self._apply_transformation(df, config.transformation)
            
            result[name] = df
        
        return result
    
    def _apply_transformation(
        self, 
        df: pd.DataFrame, 
        transformation: str
    ) -> pd.DataFrame:
        """Apply mathematical transformation to series."""
        if transformation == 'log':
            df['value'] = np.log(df['value'])
        elif transformation == 'diff':
            df['value'] = df['value'].diff()
        elif transformation == 'pct_change':
            df['value'] = df['value'].pct_change()
        
        return df
    
    def _align_and_merge(
        self, 
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Align multiple series to common date range and merge."""
        if not data:
            return pd.DataFrame()
        
        # Filter out empty DataFrames
        valid_data = {name: df for name, df in data.items() if not df.empty}
        
        if not valid_data:
            logger.warning("No valid data to merge")
            return pd.DataFrame()
        
        # Find common date range
        date_ranges = [df.index for df in valid_data.values()]
        common_start = max(dr.min() for dr in date_ranges)
        common_end = min(dr.max() for dr in date_ranges)
        
        logger.info(f"Common date range: {common_start} to {common_end}")
        
        # Merge all series
        merged = pd.DataFrame()
        for name, df in valid_data.items():
            # Filter to common date range
            df_filtered = df.loc[common_start:common_end]
            
            # Rename column
            df_filtered = df_filtered.rename(columns={'value': name})
            
            # Merge
            if merged.empty:
                merged = df_filtered
            else:
                merged = merged.join(df_filtered, how='outer')
        
        # Sort by date
        merged = merged.sort_index()
        
        return merged
    
    def _create_metadata(
        self, 
        series_config: Dict[str, SeriesConfig],
        data: pd.DataFrame
    ) -> DatasetMetadata:
        """Create metadata for versioned dataset."""
        missing_values = {}
        for col in data.columns:
            missing_count = data[col].isna().sum()
            if missing_count > 0:
                missing_values[col] = int(missing_count)
        
        return DatasetMetadata(
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
            fetch_date=datetime.now(),
            series_ids=[config.series_id for config in series_config.values()],
            start_date=str(data.index.min().date()),
            end_date=str(data.index.max().date()),
            frequency=settings.data_frequency,
            n_observations=len(data),
            missing_values=missing_values
        )
    
    def _save_versioned_dataset(
        self, 
        data: pd.DataFrame, 
        metadata: DatasetMetadata
    ):
        """Save dataset with version metadata."""
        version_dir = self.output_dir / metadata.version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_path = version_dir / "data.csv"
        data.to_csv(data_path)
        
        # Save metadata
        metadata_path = version_dir / "metadata.json"
        metadata_dict = {
            'version': metadata.version,
            'fetch_date': metadata.fetch_date.isoformat(),
            'series_ids': metadata.series_ids,
            'start_date': metadata.start_date,
            'end_date': metadata.end_date,
            'frequency': metadata.frequency,
            'n_observations': metadata.n_observations,
            'missing_values': metadata.missing_values
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.info(f"Saved versioned dataset: {metadata.version}")
    
    def load_latest_version(self) -> Optional[pd.DataFrame]:
        """Load the most recent versioned dataset."""
        versions = sorted([d for d in self.output_dir.iterdir() if d.is_dir()])
        
        if not versions:
            logger.warning("No versioned datasets found")
            return None
        
        latest_version = versions[-1]
        data_path = latest_version / "data.csv"
        
        if not data_path.exists():
            logger.warning(f"Data file not found in {latest_version}")
            return None
        
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded dataset version: {latest_version.name}")
        
        return data
