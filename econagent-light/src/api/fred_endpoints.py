"""
FastAPI endpoints for FRED data integration.
Provides REST API for accessing economic data and calibration services.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio

from ..data_integration.fred_client import FREDClient, EconomicSnapshot
from ..data_integration.calibration_engine import CalibrationEngine, CalibrationResult

logger = logging.getLogger(__name__)

# Initialize FRED client and calibration engine
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv()

from config import DEFAULT_CONFIG

# Ensure API key is loaded
fred_api_key = os.getenv("FRED_API_KEY")
if not fred_api_key:
    logger.warning("FRED_API_KEY not found in environment, checking config...")
    fred_api_key = DEFAULT_CONFIG.fred.api_key

if fred_api_key:
    logger.info(f"FRED API key loaded: {fred_api_key[:8]}...")
else:
    logger.error("No FRED API key found! Please set FRED_API_KEY in .env file")

fred_client = FREDClient(
    api_key=fred_api_key,
    cache_dir=DEFAULT_CONFIG.fred.cache_dir,
    cache_hours=DEFAULT_CONFIG.fred.cache_hours
)
calibration_engine = CalibrationEngine(fred_client)

# Create router
router = APIRouter(prefix="/api/fred", tags=["FRED Data"])

# Pydantic models for API responses
class EconomicSnapshotResponse(BaseModel):
    """Response model for economic snapshot."""
    timestamp: str
    unemployment_rate: float
    inflation_rate: float
    fed_funds_rate: float
    gdp_growth: float
    wage_growth: float
    labor_participation: float
    consumer_sentiment: Optional[float] = None

class SeriesDataResponse(BaseModel):
    """Response model for FRED series data."""
    series_id: str
    title: str
    observations: int
    start_date: str
    end_date: str
    data: List[Dict[str, Any]]

class CalibrationResponse(BaseModel):
    """Response model for calibration results."""
    unemployment_target: float
    inflation_target: float
    natural_interest_rate: float
    productivity_growth: float
    wage_adjustment_rate: float
    price_adjustment_rate: float
    calibration_date: str
    confidence_score: float
    data_period: str

class CalibrationRequest(BaseModel):
    """Request model for calibration."""
    force_recalibrate: bool = Field(False, description="Force recalibration even if recent results exist")
    historical_years: Optional[int] = Field(5, description="Years of historical data to use", ge=1, le=10)
    base_config: Optional[Dict[str, float]] = Field(None, description="Base configuration to adjust")

@router.get("/health", summary="Check FRED API health")
async def check_fred_health():
    """Check if FRED API is accessible."""
    try:
        # Test with a simple request
        test_data = fred_client.get_series('GDP', start_date='2023-01-01', end_date='2023-12-31')
        
        stats = fred_client.get_statistics()
        
        return {
            "status": "healthy",
            "fred_api": "accessible",
            "test_data_points": len(test_data),
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"FRED health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"FRED API unavailable: {str(e)}")

@router.get("/current", response_model=EconomicSnapshotResponse, summary="Get current economic snapshot")
async def get_current_economic_data():
    """Get current economic conditions from FRED data."""
    try:
        snapshot = fred_client.get_current_economic_snapshot()
        
        return EconomicSnapshotResponse(
            timestamp=snapshot.timestamp.isoformat(),
            unemployment_rate=snapshot.unemployment_rate,
            inflation_rate=snapshot.inflation_rate,
            fed_funds_rate=snapshot.fed_funds_rate,
            gdp_growth=snapshot.gdp_growth,
            wage_growth=snapshot.wage_growth,
            labor_participation=snapshot.labor_participation,
            consumer_sentiment=snapshot.consumer_sentiment
        )
    except Exception as e:
        logger.error(f"Failed to get current economic data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch economic data: {str(e)}")

@router.get("/series/{series_id}", response_model=SeriesDataResponse, summary="Get FRED series data")
async def get_fred_series(
    series_id: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    frequency: Optional[str] = Query(None, description="Data frequency (d/w/m/q/a)"),
    use_cache: bool = Query(True, description="Use cached data if available")
):
    """Get data for a specific FRED series."""
    try:
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # Fetch series data
        df = fred_client.get_series(
            series_id=series_id,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            use_cache=use_cache
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for series {series_id}")
        
        # Get series metadata
        series_info = fred_client.get_series_info(series_id)
        
        # Convert DataFrame to list of dictionaries
        data_points = []
        for date, row in df.iterrows():
            data_points.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": float(row.iloc[0]) if not pd.isna(row.iloc[0]) else None
            })
        
        return SeriesDataResponse(
            series_id=series_id,
            title=series_info.get('title', series_id),
            observations=len(data_points),
            start_date=start_date,
            end_date=end_date,
            data=data_points
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get series {series_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch series data: {str(e)}")

@router.get("/core-data", summary="Get core economic indicators")
async def get_core_economic_data(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    use_cache: bool = Query(True, description="Use cached data if available")
):
    """Get all core economic indicators needed for simulation."""
    try:
        # Set default dates
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")  # 2 years default
        
        # Fetch core economic data
        core_data = fred_client.get_core_economic_data(
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache
        )
        
        # Convert to API response format
        response_data = {}
        for series_name, df in core_data.items():
            if not df.empty:
                data_points = []
                for date, row in df.iterrows():
                    data_points.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "value": float(row.iloc[0]) if not pd.isna(row.iloc[0]) else None
                    })
                
                response_data[series_name] = {
                    "series_id": df.columns[0],
                    "observations": len(data_points),
                    "data": data_points
                }
        
        return {
            "period": f"{start_date} to {end_date}",
            "series_count": len(response_data),
            "data": response_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get core economic data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch core data: {str(e)}")

@router.post("/calibrate", response_model=CalibrationResponse, summary="Calibrate simulation parameters")
async def calibrate_simulation_parameters(request: CalibrationRequest):
    """Calibrate economic simulation parameters using FRED data."""
    try:
        logger.info(f"Starting calibration with request: {request}")
        
        # Update calibration config if provided
        if request.historical_years:
            calibration_engine.config.historical_years = request.historical_years
        
        # Perform calibration
        result = calibration_engine.calibrate_simulation_parameters(
            base_config=request.base_config,
            force_recalibrate=request.force_recalibrate
        )
        
        return CalibrationResponse(
            unemployment_target=result.unemployment_target,
            inflation_target=result.inflation_target,
            natural_interest_rate=result.natural_interest_rate,
            productivity_growth=result.productivity_growth,
            wage_adjustment_rate=result.wage_adjustment_rate,
            price_adjustment_rate=result.price_adjustment_rate,
            calibration_date=result.calibration_date.isoformat(),
            confidence_score=result.confidence_score,
            data_period=result.data_period
        )
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")

@router.get("/calibration/summary", summary="Get calibration summary")
async def get_calibration_summary():
    """Get summary of the last calibration performed."""
    try:
        summary = calibration_engine.get_calibration_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get calibration summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@router.get("/statistics", summary="Get FRED client statistics")
async def get_fred_statistics():
    """Get usage statistics for the FRED client."""
    try:
        stats = fred_client.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.post("/cache/clear", summary="Clear FRED data cache")
async def clear_fred_cache(
    older_than_hours: Optional[int] = Query(None, description="Only clear files older than this many hours")
):
    """Clear cached FRED data files."""
    try:
        cleared_count = fred_client.clear_cache(older_than_hours=older_than_hours)
        return {
            "status": "success",
            "files_cleared": cleared_count,
            "message": f"Cleared {cleared_count} cache files"
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.get("/search", summary="Search FRED series")
async def search_fred_series(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results", ge=1, le=50)
):
    """Search for FRED series by text query."""
    try:
        results = fred_client.search_series(search_text=query, limit=limit)
        
        return {
            "query": query,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/indicators/dashboard", summary="Get dashboard economic indicators")
async def get_dashboard_indicators():
    """Get key economic indicators formatted for dashboard display."""
    try:
        # Get current snapshot
        snapshot = fred_client.get_current_economic_snapshot()
        
        # Get recent historical data for trends
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")  # 3 months
        
        # Fetch trend data
        unemployment_data = fred_client.get_series('UNRATE', start_date=start_date, end_date=end_date)
        inflation_data = fred_client.get_series('CPIAUCSL', start_date=start_date, end_date=end_date)
        
        # Calculate trends
        unemployment_trend = "stable"
        inflation_trend = "stable"
        
        if len(unemployment_data) >= 2:
            recent_unemployment = unemployment_data['UNRATE'].iloc[-1]
            previous_unemployment = unemployment_data['UNRATE'].iloc[-2]
            if recent_unemployment > previous_unemployment + 0.1:
                unemployment_trend = "rising"
            elif recent_unemployment < previous_unemployment - 0.1:
                unemployment_trend = "falling"
        
        if len(inflation_data) >= 12:  # Need year-over-year for inflation
            current_cpi = inflation_data['CPIAUCSL'].iloc[-1]
            year_ago_cpi = inflation_data['CPIAUCSL'].iloc[-12] if len(inflation_data) >= 12 else current_cpi
            current_inflation = ((current_cpi - year_ago_cpi) / year_ago_cpi) * 100
            
            if current_inflation > 3.0:
                inflation_trend = "rising"
            elif current_inflation < 1.0:
                inflation_trend = "falling"
        
        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "indicators": {
                "unemployment": {
                    "value": snapshot.unemployment_rate,
                    "trend": unemployment_trend,
                    "unit": "percent"
                },
                "inflation": {
                    "value": snapshot.inflation_rate,
                    "trend": inflation_trend,
                    "unit": "percent"
                },
                "fed_funds_rate": {
                    "value": snapshot.fed_funds_rate,
                    "trend": "stable",
                    "unit": "percent"
                },
                "wage_growth": {
                    "value": snapshot.wage_growth,
                    "trend": "stable",
                    "unit": "percent"
                },
                "labor_participation": {
                    "value": snapshot.labor_participation,
                    "trend": "stable",
                    "unit": "percent"
                }
            },
            "summary": {
                "economic_health": "moderate" if 3.0 <= snapshot.unemployment_rate <= 6.0 else "concerning",
                "inflation_status": "target" if 1.5 <= snapshot.inflation_rate <= 2.5 else "off_target",
                "policy_stance": "neutral" if 2.0 <= snapshot.fed_funds_rate <= 4.0 else "accommodative" if snapshot.fed_funds_rate < 2.0 else "restrictive"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard indicators: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get indicators: {str(e)}")

import pandas as pd