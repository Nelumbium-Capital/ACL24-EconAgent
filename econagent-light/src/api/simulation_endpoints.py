"""
FastAPI endpoints for simulation management.
Provides REST API for creating, running, and monitoring economic simulations.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
import uuid

from ..data_integration.fred_client import FREDClient
from ..data_integration.calibration_engine import CalibrationEngine

logger = logging.getLogger(__name__)

# Initialize clients
fred_client = FREDClient()
calibration_engine = CalibrationEngine(fred_client)

# Create router
router = APIRouter(prefix="/api/simulations", tags=["Simulations"])

# In-memory storage for simulation status (in production, use a database)
simulation_store: Dict[str, Dict[str, Any]] = {}
simulation_results: Dict[str, Dict[str, Any]] = {}

# Pydantic models
class SimulationConfig(BaseModel):
    """Configuration for creating a new simulation."""
    name: str = Field(..., description="Simulation name")
    num_agents: int = Field(100, ge=10, le=1000, description="Number of agents")
    num_years: int = Field(20, ge=1, le=50, description="Simulation duration in years")
    use_fred_calibration: bool = Field(True, description="Use FRED data for calibration")
    fred_start_date: Optional[str] = Field(None, description="FRED data start date (YYYY-MM-DD)")
    economic_scenario: str = Field("baseline", description="Economic scenario")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    # Advanced parameters
    productivity: Optional[float] = Field(1.0, ge=0.1, le=5.0, description="Base productivity")
    skill_change: Optional[float] = Field(0.02, ge=0.0, le=0.1, description="Skill change rate")
    price_change: Optional[float] = Field(0.02, ge=0.0, le=0.1, description="Price change rate")

class SimulationStatus(BaseModel):
    """Current simulation status and progress."""
    simulation_id: str
    name: str
    status: str  # "created", "running", "completed", "failed"
    current_step: int
    total_steps: int
    progress_percent: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None

class SimulationResult(BaseModel):
    """Simulation results summary."""
    simulation_id: str
    name: str
    config: SimulationConfig
    status: str
    duration_seconds: Optional[float] = None
    final_metrics: Optional[Dict[str, float]] = None
    economic_indicators: Optional[Dict[str, List[float]]] = None
    agent_statistics: Optional[Dict[str, Any]] = None

@router.post("/", response_model=Dict[str, str], summary="Create new simulation")
async def create_simulation(config: SimulationConfig):
    """Create a new economic simulation with the given configuration."""
    try:
        # Generate unique simulation ID
        simulation_id = str(uuid.uuid4())
        
        # Validate configuration
        if config.num_years * 12 > 600:  # Limit to 50 years max
            raise HTTPException(status_code=400, detail="Simulation too long (max 50 years)")
        
        # Store simulation configuration
        simulation_data = {
            "id": simulation_id,
            "name": config.name,
            "config": config.dict(),
            "status": "created",
            "current_step": 0,
            "total_steps": config.num_years * 12,  # Monthly steps
            "progress_percent": 0.0,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "current_metrics": None,
            "error_message": None
        }
        
        simulation_store[simulation_id] = simulation_data
        
        logger.info(f"Created simulation {simulation_id}: {config.name}")
        
        return {
            "simulation_id": simulation_id,
            "status": "created",
            "message": f"Simulation '{config.name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create simulation: {str(e)}")

@router.post("/{simulation_id}/start", summary="Start simulation")
async def start_simulation(simulation_id: str, background_tasks: BackgroundTasks):
    """Start a created simulation."""
    if simulation_id not in simulation_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    simulation_data = simulation_store[simulation_id]
    
    if simulation_data["status"] != "created":
        raise HTTPException(status_code=400, detail=f"Simulation is {simulation_data['status']}, cannot start")
    
    try:
        # Update status to running
        simulation_data["status"] = "running"
        simulation_data["started_at"] = datetime.now().isoformat()
        
        # Start simulation in background
        background_tasks.add_task(run_simulation_task, simulation_id)
        
        logger.info(f"Started simulation {simulation_id}")
        
        return {
            "simulation_id": simulation_id,
            "status": "running",
            "message": "Simulation started successfully"
        }
        
    except Exception as e:
        simulation_data["status"] = "failed"
        simulation_data["error_message"] = str(e)
        logger.error(f"Failed to start simulation {simulation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start simulation: {str(e)}")

@router.get("/{simulation_id}/status", response_model=SimulationStatus, summary="Get simulation status")
async def get_simulation_status(simulation_id: str):
    """Get the current status of a simulation."""
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
        current_metrics=simulation_data.get("current_metrics"),
        error_message=simulation_data.get("error_message")
    )

@router.get("/{simulation_id}/results", response_model=SimulationResult, summary="Get simulation results")
async def get_simulation_results(simulation_id: str):
    """Get the results of a completed simulation."""
    if simulation_id not in simulation_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    simulation_data = simulation_store[simulation_id]
    
    if simulation_data["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Simulation not completed yet")
    
    # Get results from storage
    results = simulation_results.get(simulation_id, {})
    
    # Calculate duration
    duration_seconds = None
    if simulation_data.get("started_at") and simulation_data.get("completed_at"):
        start_time = datetime.fromisoformat(simulation_data["started_at"])
        end_time = datetime.fromisoformat(simulation_data["completed_at"])
        duration_seconds = (end_time - start_time).total_seconds()
    
    return SimulationResult(
        simulation_id=simulation_id,
        name=simulation_data["name"],
        config=SimulationConfig(**simulation_data["config"]),
        status=simulation_data["status"],
        duration_seconds=duration_seconds,
        final_metrics=results.get("final_metrics"),
        economic_indicators=results.get("economic_indicators"),
        agent_statistics=results.get("agent_statistics")
    )

@router.get("/", summary="List all simulations")
async def list_simulations(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results")
):
    """List all simulations with optional status filter."""
    simulations = []
    
    for sim_id, sim_data in simulation_store.items():
        if status and sim_data["status"] != status:
            continue
            
        simulations.append({
            "simulation_id": sim_id,
            "name": sim_data["name"],
            "status": sim_data["status"],
            "created_at": sim_data["created_at"],
            "progress_percent": sim_data["progress_percent"]
        })
    
    # Sort by creation date (newest first)
    simulations.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "simulations": simulations[:limit],
        "total_count": len(simulations)
    }

@router.delete("/{simulation_id}", summary="Delete simulation")
async def delete_simulation(simulation_id: str):
    """Delete a simulation and its results."""
    if simulation_id not in simulation_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    simulation_data = simulation_store[simulation_id]
    
    # Don't delete running simulations
    if simulation_data["status"] == "running":
        raise HTTPException(status_code=400, detail="Cannot delete running simulation")
    
    # Remove from storage
    del simulation_store[simulation_id]
    if simulation_id in simulation_results:
        del simulation_results[simulation_id]
    
    logger.info(f"Deleted simulation {simulation_id}")
    
    return {"message": "Simulation deleted successfully"}

@router.post("/{simulation_id}/stop", summary="Stop running simulation")
async def stop_simulation(simulation_id: str):
    """Stop a running simulation."""
    if simulation_id not in simulation_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    simulation_data = simulation_store[simulation_id]
    
    if simulation_data["status"] != "running":
        raise HTTPException(status_code=400, detail="Simulation is not running")
    
    # Update status (in a real implementation, you'd need to actually stop the background task)
    simulation_data["status"] = "stopped"
    simulation_data["completed_at"] = datetime.now().isoformat()
    
    logger.info(f"Stopped simulation {simulation_id}")
    
    return {"message": "Simulation stopped successfully"}

@router.get("/{simulation_id}/export", summary="Export simulation results")
async def export_simulation_results(
    simulation_id: str,
    format: str = Query("json", description="Export format (json, csv)")
):
    """Export simulation results in the specified format."""
    if simulation_id not in simulation_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    simulation_data = simulation_store[simulation_id]
    
    if simulation_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Simulation not completed")
    
    results = simulation_results.get(simulation_id, {})
    
    if format.lower() == "json":
        return results
    elif format.lower() == "csv":
        # In a real implementation, convert to CSV format
        raise HTTPException(status_code=501, detail="CSV export not implemented yet")
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")

# Background task for running simulations
async def run_simulation_task(simulation_id: str):
    """Background task to run a simulation."""
    try:
        simulation_data = simulation_store[simulation_id]
        config = SimulationConfig(**simulation_data["config"])
        
        logger.info(f"Running simulation {simulation_id}: {config.name}")
        
        # Get FRED calibration if requested
        calibration_result = None
        if config.use_fred_calibration:
            try:
                calibration_result = calibration_engine.calibrate_simulation_parameters()
                logger.info(f"Calibration completed for {simulation_id}")
            except Exception as e:
                logger.warning(f"Calibration failed for {simulation_id}: {e}")
        
        # Simulate the simulation (mock implementation)
        total_steps = config.num_years * 12
        
        # Mock economic indicators
        unemployment_rates = []
        inflation_rates = []
        gdp_growth = []
        
        for step in range(total_steps):
            # Update progress
            simulation_data["current_step"] = step + 1
            simulation_data["progress_percent"] = ((step + 1) / total_steps) * 100
            
            # Mock economic calculations
            base_unemployment = calibration_result.unemployment_target if calibration_result else 5.0
            base_inflation = calibration_result.inflation_target if calibration_result else 2.0
            
            # Add some random variation
            import random
            unemployment = base_unemployment + random.uniform(-1.0, 1.0)
            inflation = base_inflation + random.uniform(-0.5, 0.5)
            gdp = 2.5 + random.uniform(-1.0, 1.0)
            
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
            await asyncio.sleep(0.1)  # 100ms per step for demo
        
        # Simulation completed
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
                "gdp_growth": gdp_growth
            },
            "agent_statistics": {
                "num_agents": config.num_agents,
                "simulation_years": config.num_years
            }
        }
        
        logger.info(f"Simulation {simulation_id} completed successfully")
        
    except Exception as e:
        # Simulation failed
        simulation_data["status"] = "failed"
        simulation_data["error_message"] = str(e)
        simulation_data["completed_at"] = datetime.now().isoformat()
        
        logger.error(f"Simulation {simulation_id} failed: {e}")