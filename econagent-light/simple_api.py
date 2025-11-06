#!/usr/bin/env python3
"""
Simple FastAPI server for EconAgent-Light frontend.
Provides basic endpoints without complex dependencies.
"""

import logging
import asyncio
import json
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="EconAgent-Light API",
    description="Economic simulation platform",
    version="1.0.0",
    docs_url="/api/docs"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
simulation_store: Dict[str, Dict[str, Any]] = {}
simulation_results: Dict[str, Dict[str, Any]] = {}

# Pydantic models
class SimulationConfig(BaseModel):
    name: str = Field(..., description="Simulation name")
    num_agents: int = Field(100, ge=10, le=1000, description="Number of agents")
    num_years: int = Field(20, ge=1, le=50, description="Simulation duration in years")
    use_fred_calibration: bool = Field(True, description="Use FRED data for calibration")
    random_seed: Optional[int] = Field(None, description="Random seed")

class SimulationStatus(BaseModel):
    simulation_id: str
    name: str
    status: str
    current_step: int
    total_steps: int
    progress_percent: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_metrics: Optional[Dict[str, float]] = None

# Health check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "EconAgent-Light API"}

# FRED endpoints (mock data)
@app.get("/api/fred/current")
async def get_current_economic_data():
    """Mock current economic data."""
    return {
        "timestamp": datetime.now().isoformat(),
        "unemployment_rate": 3.7 + random.uniform(-0.5, 0.5),
        "inflation_rate": 2.1 + random.uniform(-0.3, 0.3),
        "fed_funds_rate": 5.25 + random.uniform(-0.25, 0.25),
        "gdp_growth": 2.4 + random.uniform(-0.5, 0.5),
        "wage_growth": 4.2 + random.uniform(-0.3, 0.3),
        "labor_participation": 63.4 + random.uniform(-0.2, 0.2)
    }

@app.get("/api/fred/indicators/dashboard")
async def get_dashboard_indicators():
    """Mock dashboard indicators."""
    unemployment = 3.7 + random.uniform(-0.5, 0.5)
    inflation = 2.1 + random.uniform(-0.3, 0.3)
    fed_funds = 5.25 + random.uniform(-0.25, 0.25)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "indicators": {
            "unemployment": {
                "value": unemployment,
                "trend": "stable",
                "unit": "percent"
            },
            "inflation": {
                "value": inflation,
                "trend": "stable",
                "unit": "percent"
            },
            "fed_funds_rate": {
                "value": fed_funds,
                "trend": "stable",
                "unit": "percent"
            }
        },
        "summary": {
            "economic_health": "moderate",
            "inflation_status": "target",
            "policy_stance": "neutral"
        }
    }

# Simulation endpoints
@app.post("/api/simulations/")
async def create_simulation(config: SimulationConfig):
    """Create a new simulation."""
    simulation_id = str(uuid.uuid4())
    
    simulation_data = {
        "id": simulation_id,
        "name": config.name,
        "config": config.dict(),
        "status": "created",
        "current_step": 0,
        "total_steps": config.num_years * 12,
        "progress_percent": 0.0,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "current_metrics": None
    }
    
    simulation_store[simulation_id] = simulation_data
    
    return {
        "simulation_id": simulation_id,
        "status": "created",
        "message": f"Simulation '{config.name}' created successfully"
    }

@app.post("/api/simulations/{simulation_id}/start")
async def start_simulation(simulation_id: str, background_tasks: BackgroundTasks):
    """Start a simulation."""
    if simulation_id not in simulation_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    simulation_data = simulation_store[simulation_id]
    
    if simulation_data["status"] != "created":
        raise HTTPException(status_code=400, detail="Simulation already started")
    
    simulation_data["status"] = "running"
    simulation_data["started_at"] = datetime.now().isoformat()
    
    background_tasks.add_task(run_simulation_task, simulation_id)
    
    return {"simulation_id": simulation_id, "status": "running"}

@app.get("/api/simulations/{simulation_id}/status", response_model=SimulationStatus)
async def get_simulation_status(simulation_id: str):
    """Get simulation status."""
    if simulation_id not in simulation_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    simulation_data = simulation_store[simulation_id]
    
    return SimulationStatus(
        simulation_id=simulation_id,
        name=simulation_data["name"],
        status=simulation_data["status"],
        current_step=simulation_data["current_step"],
        total_steps=simulation_data["total_steps"],
        progress_percent=simulation_data["progress_percent"],
        created_at=simulation_data["created_at"],
        started_at=simulation_data.get("started_at"),
        completed_at=simulation_data.get("completed_at"),
        current_metrics=simulation_data.get("current_metrics")
    )

@app.get("/api/simulations/{simulation_id}/results")
async def get_simulation_results(simulation_id: str):
    """Get simulation results."""
    if simulation_id not in simulation_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    simulation_data = simulation_store[simulation_id]
    
    if simulation_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Simulation not completed")
    
    return simulation_results.get(simulation_id, {})

@app.get("/api/simulations/")
async def list_simulations():
    """List all simulations."""
    simulations = []
    
    for sim_id, sim_data in simulation_store.items():
        simulations.append({
            "simulation_id": sim_id,
            "name": sim_data["name"],
            "status": sim_data["status"],
            "created_at": sim_data["created_at"],
            "progress_percent": sim_data["progress_percent"]
        })
    
    simulations.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {"simulations": simulations, "total_count": len(simulations)}

async def run_simulation_task(simulation_id: str):
    """Background task to run simulation."""
    try:
        simulation_data = simulation_store[simulation_id]
        config = SimulationConfig(**simulation_data["config"])
        
        total_steps = config.num_years * 12
        
        # Mock economic data
        unemployment_rates = []
        inflation_rates = []
        gdp_growth = []
        
        for step in range(total_steps):
            # Update progress
            simulation_data["current_step"] = step + 1
            simulation_data["progress_percent"] = ((step + 1) / total_steps) * 100
            
            # Generate mock economic data
            unemployment = 3.7 + random.uniform(-2.0, 2.0)
            inflation = 2.1 + random.uniform(-1.0, 1.0)
            gdp = 2.5 + random.uniform(-1.5, 1.5)
            
            unemployment_rates.append(max(0, unemployment))
            inflation_rates.append(inflation)
            gdp_growth.append(gdp)
            
            # Update current metrics
            simulation_data["current_metrics"] = {
                "unemployment_rate": unemployment,
                "inflation_rate": inflation,
                "gdp_growth": gdp,
                "step": step + 1
            }
            
            # Simulate processing time
            await asyncio.sleep(0.05)  # 50ms per step
        
        # Complete simulation
        simulation_data["status"] = "completed"
        simulation_data["completed_at"] = datetime.now().isoformat()
        simulation_data["progress_percent"] = 100.0
        
        # Store results
        simulation_results[simulation_id] = {
            "final_metrics": {
                "avg_unemployment": sum(unemployment_rates) / len(unemployment_rates),
                "avg_inflation": sum(inflation_rates) / len(inflation_rates),
                "avg_gdp_growth": sum(gdp_growth) / len(gdp_growth),
                "total_steps": total_steps
            },
            "economic_indicators": {
                "unemployment_rates": unemployment_rates,
                "inflation_rates": inflation_rates,
                "gdp_growth": gdp_growth,
                "months": list(range(1, total_steps + 1))
            },
            "agent_statistics": {
                "num_agents": config.num_agents,
                "simulation_years": config.num_years
            }
        }
        
        logger.info(f"Simulation {simulation_id} completed")
        
    except Exception as e:
        simulation_data["status"] = "failed"
        simulation_data["error_message"] = str(e)
        simulation_data["completed_at"] = datetime.now().isoformat()
        logger.error(f"Simulation {simulation_id} failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)