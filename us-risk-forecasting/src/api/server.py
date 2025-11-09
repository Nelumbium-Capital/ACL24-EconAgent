"""
FastAPI server for Risk Forecasting Dashboard.
Serves data to React frontend.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

from src.data.fred_client import FREDClient
from src.data.pipeline import DataPipeline, MissingValueHandler, FrequencyAligner
from src.data.data_models import SeriesConfig
from src.kri.calculator import KRICalculator
from src.kri.definitions import kri_registry
from src.models.llm_forecaster import LLMEnsembleForecaster
from src.utils.logging_config import logger
from config import settings

app = FastAPI(title="Risk Forecasting API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache
data_cache = {
    'economic_data': None,
    'forecasts': None,
    'kris': None,
    'risk_levels': None,
    'last_update': None
}


def fetch_all_data():
    """Fetch and process all data."""
    logger.info("Fetching data for API...")
    
    fred_client = FREDClient()
    pipeline = DataPipeline(fred_client)
    pipeline.add_transformer(MissingValueHandler(method='ffill'))
    pipeline.add_transformer(FrequencyAligner(target_frequency='M'))
    
    series_config = {
        'unemployment': SeriesConfig(
            series_id='UNRATE', name='Unemployment Rate',
            start_date='2018-01-01', end_date='2024-01-01', frequency='monthly'
        ),
        'inflation': SeriesConfig(
            series_id='CPIAUCSL', name='CPI Inflation',
            start_date='2018-01-01', end_date='2024-01-01', frequency='monthly',
            transformation='pct_change'
        ),
        'interest_rate': SeriesConfig(
            series_id='FEDFUNDS', name='Federal Funds Rate',
            start_date='2018-01-01', end_date='2024-01-01', frequency='monthly'
        ),
        'credit_spread': SeriesConfig(
            series_id='BAA10Y', name='BAA-Treasury Spread',
            start_date='2018-01-01', end_date='2024-01-01', frequency='monthly'
        )
    }
    
    economic_data = pipeline.process(series_config)
    
    # Generate forecasts
    forecast_horizon = 12
    forecasts_dict = {}
    llm_forecaster = LLMEnsembleForecaster()
    
    for col in economic_data.columns:
        series = economic_data[col].dropna().values
        try:
            result = llm_forecaster.forecast(
                series=series, horizon=forecast_horizon,
                series_name=col, use_llm=False
            )
            forecasts_dict[col] = result['ensemble']
        except:
            forecasts_dict[col] = np.full(forecast_horizon, series[-1])
    
    forecast_dates = pd.date_range(
        start=economic_data.index[-1] + pd.DateOffset(months=1),
        periods=forecast_horizon, freq='ME'
    )
    forecasts_df = pd.DataFrame(forecasts_dict, index=forecast_dates)
    
    # Compute KRIs
    kri_calc = KRICalculator()
    combined_data = pd.concat([economic_data.tail(12), forecasts_df])
    kris = kri_calc.compute_all_kris(forecasts=combined_data)
    risk_levels = kri_calc.evaluate_thresholds(kris)
    
    data_cache['economic_data'] = economic_data
    data_cache['forecasts'] = forecasts_df
    data_cache['kris'] = kris
    data_cache['risk_levels'] = risk_levels
    data_cache['last_update'] = datetime.now()
    
    logger.info("Data updated successfully")


@app.on_event("startup")
async def startup_event():
    """Initialize data on startup."""
    fetch_all_data()


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/dashboard/summary")
async def get_dashboard_summary():
    """Get dashboard summary with KPIs."""
    if data_cache['economic_data'] is None:
        fetch_all_data()
    
    econ_data = data_cache['economic_data']
    latest = econ_data.iloc[-1]
    
    return {
        "timestamp": datetime.now().isoformat(),
        "kpis": {
            "unemployment": {
                "value": float(latest['unemployment']),
                "unit": "%",
                "change": float(latest['unemployment'] - econ_data.iloc[-2]['unemployment']),
                "trend": "down" if latest['unemployment'] < econ_data.iloc[-2]['unemployment'] else "up"
            },
            "inflation": {
                "value": float(latest['inflation'] * 100),
                "unit": "%",
                "change": float((latest['inflation'] - econ_data.iloc[-2]['inflation']) * 100),
                "trend": "down" if latest['inflation'] < econ_data.iloc[-2]['inflation'] else "up"
            },
            "interest_rate": {
                "value": float(latest['interest_rate']),
                "unit": "%",
                "change": float(latest['interest_rate'] - econ_data.iloc[-2]['interest_rate']),
                "trend": "down" if latest['interest_rate'] < econ_data.iloc[-2]['interest_rate'] else "up"
            },
            "credit_spread": {
                "value": float(latest['credit_spread']),
                "unit": "%",
                "change": float(latest['credit_spread'] - econ_data.iloc[-2]['credit_spread']),
                "trend": "down" if latest['credit_spread'] < econ_data.iloc[-2]['credit_spread'] else "up"
            }
        },
        "risk_summary": {
            "critical": sum(1 for v in data_cache['risk_levels'].values() if v.value == 'critical'),
            "high": sum(1 for v in data_cache['risk_levels'].values() if v.value == 'high'),
            "medium": sum(1 for v in data_cache['risk_levels'].values() if v.value == 'medium'),
            "low": sum(1 for v in data_cache['risk_levels'].values() if v.value == 'low')
        }
    }


@app.get("/api/economic-data")
async def get_economic_data():
    """Get historical economic data."""
    if data_cache['economic_data'] is None:
        fetch_all_data()
    
    df = data_cache['economic_data']
    
    return {
        "data": [
            {
                "date": idx.strftime('%Y-%m-%d'),
                "unemployment": float(row['unemployment']),
                "inflation": float(row['inflation'] * 100),
                "interest_rate": float(row['interest_rate']),
                "credit_spread": float(row['credit_spread'])
            }
            for idx, row in df.iterrows()
        ]
    }


@app.get("/api/forecasts")
async def get_forecasts():
    """Get forecast data."""
    if data_cache['forecasts'] is None:
        fetch_all_data()
    
    df = data_cache['forecasts']
    
    return {
        "data": [
            {
                "date": idx.strftime('%Y-%m-%d'),
                "unemployment": float(row['unemployment']),
                "inflation": float(row['inflation'] * 100),
                "interest_rate": float(row['interest_rate']),
                "credit_spread": float(row['credit_spread'])
            }
            for idx, row in df.iterrows()
        ]
    }


@app.get("/api/kris")
async def get_kris():
    """Get all KRIs with risk levels."""
    if data_cache['kris'] is None:
        fetch_all_data()
    
    kris = data_cache['kris']
    risk_levels = data_cache['risk_levels']
    
    result = []
    for kri_name, value in kris.items():
        kri_def = kri_registry.get_kri(kri_name)
        result.append({
            "name": kri_name,
            "display_name": kri_name.replace('_', ' ').title(),
            "value": float(value),
            "unit": kri_def.unit,
            "category": kri_def.category.value,
            "risk_level": risk_levels[kri_name].value,
            "is_leading": kri_def.is_leading,
            "description": kri_def.description,
            "thresholds": kri_def.thresholds
        })
    
    return {"kris": result}


@app.post("/api/refresh")
async def refresh_data():
    """Refresh all data from FRED."""
    try:
        fetch_all_data()
        return {
            "status": "success",
            "timestamp": data_cache['last_update'].isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to refresh data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Risk Forecasting API server...")
    logger.info("API will be available at: http://localhost:8000")
    logger.info("API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
