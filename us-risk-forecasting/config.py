"""
Configuration management for US Financial Risk Forecasting System.
"""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    fred_api_key: str
    wrds_username: Optional[str] = None
    wrds_password: Optional[str] = None
    
    # Data Settings
    data_cache_dir: Path = Path("data/cache")
    data_start_date: str = "2015-01-01"
    data_frequency: str = "monthly"
    
    # Model Settings
    forecast_horizon: int = 12
    ensemble_weights_method: str = "performance"
    retrain_frequency: str = "weekly"
    
    # Simulation Settings
    n_banks: int = 10
    n_firms: int = 50
    simulation_steps: int = 100
    
    # Dashboard Settings
    dashboard_port: int = 8050
    auto_refresh_interval: int = 60
    
    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("logs/risk_forecasting.log")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
