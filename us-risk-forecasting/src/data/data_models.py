"""
Data models for configuration and metadata.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class SeriesConfig:
    """Configuration for fetching a time series."""
    series_id: str
    name: str
    start_date: str
    end_date: str
    frequency: str  # 'daily', 'monthly', 'quarterly'
    transformation: Optional[str] = None  # 'log', 'diff', 'pct_change', None


@dataclass
class DatasetMetadata:
    """Metadata for a versioned dataset."""
    version: str
    fetch_date: datetime
    series_ids: list
    start_date: str
    end_date: str
    frequency: str
    n_observations: int
    missing_values: dict
