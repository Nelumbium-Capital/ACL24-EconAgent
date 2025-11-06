"""
Data integration module for EconAgent-Light.
Handles real economic data from FRED and other sources.
"""

from .fred_client import FREDClient
from .data_processor import EconomicDataProcessor
from .calibration import ModelCalibrator
from .real_data_manager import RealDataManager

__all__ = ['FREDClient', 'EconomicDataProcessor', 'ModelCalibrator', 'RealDataManager']