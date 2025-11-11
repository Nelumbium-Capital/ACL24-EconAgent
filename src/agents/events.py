"""
Event schemas for the event-driven agent architecture.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskSignal:
    """Risk signal identified by Market Analyst."""
    type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    timestamp: datetime
    affected_kris: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastResult:
    """Result from a forecasting model."""
    model_name: str
    kri_name: str
    forecast_date: datetime
    horizon: int
    point_forecast: np.ndarray
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    confidence_level: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataUpdateEvent:
    """Event published when new data is available."""
    series_ids: List[str]
    data: pd.DataFrame
    timestamp: datetime
    source: str = "FRED"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastCompleteEvent:
    """Event published when forecasting completes."""
    forecasts: Dict[str, ForecastResult]
    horizon: int
    timestamp: datetime
    model_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationCompleteEvent:
    """Event published when stress test simulation completes."""
    scenario_name: str
    results: pd.DataFrame
    summary_stats: Dict[str, float]
    timestamp: datetime
    n_steps: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KRIUpdateEvent:
    """Event published when KRIs are updated."""
    kris: Dict[str, float]
    alerts: Dict[str, str]
    timestamp: datetime
    risk_levels: Dict[str, RiskLevel] = field(default_factory=dict)
    trends: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskSignalEvent:
    """Event published when risk signals are detected."""
    signals: List[RiskSignal]
    timestamp: datetime
    source_agent: str = "MarketAnalyst"
    metadata: Dict[str, Any] = field(default_factory=dict)
