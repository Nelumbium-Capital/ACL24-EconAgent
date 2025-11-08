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
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize clients with API key from environment
fred_api_key = os.getenv("FRED_API_KEY")
if fred_api_key:
    logger.info(f"FRED API key loaded for simulations: {fred_api_key[:8]}...")
else:
    logger.warning("No FRED API key found for simulations")

fred_client = FREDClient(api_key=fred_api_key)
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
    """Background task to run a simulation using the actual Mesa EconModel."""
    try:
        simulation_data = simulation_store[simulation_id]
        config = SimulationConfig(**simulation_data["config"])
        
        logger.info(f"Running REAL Mesa simulation {simulation_id}: {config.name}")
        
        # Get FRED calibration if requested
        calibration_result = None
        fred_api_key = os.getenv("FRED_API_KEY")
        
        if config.use_fred_calibration:
            try:
                calibration_result = calibration_engine.calibrate_simulation_parameters()
                logger.info(f"Calibration completed for {simulation_id}: unemployment_target={calibration_result.unemployment_target:.2f}%, inflation_target={calibration_result.inflation_target:.2f}%")
            except Exception as e:
                logger.warning(f"Calibration failed for {simulation_id}: {e}")
        
        # Import and create the actual Mesa EconModel
        from ..mesa_model.model import EconModel
        
        # Create model with configuration
        model_params = {
            'n_agents': config.num_agents,
            'episode_length': config.num_years * 12,
            'random_seed': config.random_seed,
            'productivity': config.productivity or 1.0,
            'skill_change': config.skill_change or 0.02,
            'price_change': config.price_change or 0.02,
            'fred_api_key': fred_api_key,
            'enable_real_data': config.use_fred_calibration,
            'real_data_update_frequency': 12,  # Update from FRED every 12 months
            'save_frequency': 6,
            'log_frequency': 3
        }
        
        # Apply calibration results if available
        if calibration_result:
            model_params['base_interest_rate'] = calibration_result.natural_interest_rate
            # Calibration provides targets that influence the model
            logger.info(f"Using calibrated parameters: interest_rate={calibration_result.natural_interest_rate:.3f}")
        
        logger.info(f"Creating Mesa EconModel with {config.num_agents} agents for {config.num_years} years")
        model = EconModel(**model_params)
        
        total_steps = config.num_years * 12
        
        # Run the actual Mesa simulation
        for step in range(total_steps):
            # Run one step of the Mesa model
            model.step()
            
            # Update progress
            simulation_data["current_step"] = step + 1
            simulation_data["progress_percent"] = ((step + 1) / total_steps) * 100
            
            # Get current metrics from the model
            simulation_data["current_metrics"] = {
                "unemployment_rate": model.unemployment_rate * 100,  # Convert to percentage
                "inflation_rate": model.inflation_rate * 100,
                "gdp_growth": (model._calculate_gdp() / max(model._calculate_gdp() - 100, 1)) * 100 if step > 0 else 2.5,
                "step": step + 1
            }
            
            # Log progress periodically
            if (step + 1) % 12 == 0:  # Every year
                year = (step + 1) // 12
                logger.info(f"Simulation {simulation_id} - Year {year}: "
                          f"Unemployment={model.unemployment_rate*100:.1f}%, "
                          f"Inflation={model.inflation_rate*100:.1f}%, "
                          f"GDP=${model._calculate_gdp():.0f}")
            
            # Small delay to prevent blocking
            if step % 10 == 0:
                await asyncio.sleep(0.01)
        
        # Get final results from the model
        model_data = model.get_results_dataframe()
        
        # Extract economic indicators
        unemployment_rates = (model_data['Unemployment'] * 100).tolist()
        inflation_rates = (model_data['Inflation'] * 100).tolist()
        
        # Calculate GDP growth rates
        gdp_values = model_data['GDP'].tolist()
        gdp_growth = [2.5]  # Initial value
        for i in range(1, len(gdp_values)):
            if gdp_values[i-1] > 0:
                growth = ((gdp_values[i] - gdp_values[i-1]) / gdp_values[i-1]) * 100
                gdp_growth.append(growth)
            else:
                gdp_growth.append(0.0)
        
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
                "total_steps": total_steps,
                "final_gini": model._calculate_gini(),
                "final_total_wealth": model._calculate_total_wealth()
            },
            "economic_indicators": {
                "unemployment_rates": unemployment_rates,
                "inflation_rates": inflation_rates,
                "gdp_growth": gdp_growth
            },
            "agent_statistics": {
                "num_agents": config.num_agents,
                "simulation_years": config.num_years,
                "final_employment_rate": model._calculate_employment_rate(),
                "average_consumption": model._calculate_average_consumption()
            }
        }
        
        logger.info(f"Mesa simulation {simulation_id} completed successfully: "
                   f"Avg Unemployment={simulation_results[simulation_id]['final_metrics']['avg_unemployment']:.1f}%, "
                   f"Avg Inflation={simulation_results[simulation_id]['final_metrics']['avg_inflation']:.1f}%")
        
    except Exception as e:
        # Simulation failed
        simulation_data["status"] = "failed"
        simulation_data["error_message"] = str(e)
        simulation_data["completed_at"] = datetime.now().isoformat()
        
        logger.error(f"Simulation {simulation_id} failed: {e}", exc_info=True)