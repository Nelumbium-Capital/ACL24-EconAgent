"""
FRED data initialization for Mesa economic model.
Properly initializes the model with real economic data.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def initialize_model_with_fred(model, fred_api_key: str) -> bool:
    """
    Initialize the economic model with current FRED data.
    
    Args:
        model: The EconModel instance to initialize
        fred_api_key: FRED API key
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from ..data_integration.fred_client import FREDClient
        
        logger.info("🔄 Initializing model with FRED data...")
        
        # Create FRED client
        fred_client = FREDClient(api_key=fred_api_key)
        
        # Get current economic snapshot
        snapshot = fred_client.get_current_economic_snapshot()
        
        logger.info(f"📊 FRED Data Retrieved:")
        logger.info(f"   Unemployment: {snapshot.unemployment_rate:.1f}%")
        logger.info(f"   Inflation: {snapshot.inflation_rate:.1f}%")
        logger.info(f"   Fed Funds Rate: {snapshot.fed_funds_rate:.1f}%")
        logger.info(f"   GDP Growth: {snapshot.gdp_growth:.1f}%")
        logger.info(f"   Wage Growth: {snapshot.wage_growth:.1f}%")
        
        # Initialize model with FRED data (convert percentages to decimals)
        model.unemployment_rate = snapshot.unemployment_rate / 100.0
        model.inflation_rate = snapshot.inflation_rate / 100.0
        model.interest_rate = snapshot.fed_funds_rate / 100.0
        model.base_interest_rate = model.interest_rate
        
        # Store FRED values for reference and updates
        model.real_unemployment_rate = model.unemployment_rate
        model.real_inflation_rate = model.inflation_rate
        model.real_fed_funds_rate = model.interest_rate
        model.real_gdp_growth = snapshot.gdp_growth / 100.0
        model.real_wage_growth = snapshot.wage_growth / 100.0
        
        # Keep price and wage at normalized 1.0 scale
        # The inflation_rate tracks the change rate
        model.goods_price = 1.0
        model.average_wage = 1.0
        
        # Initialize inventory at equilibrium
        # Target: enough for ~50% of one month's consumption
        avg_skill = model.payment_max_skill_multiplier / 2
        employment_rate = 1.0 - model.unemployment_rate
        monthly_production = (model.n_agents * model.labor_hours * avg_skill * 
                            model.productivity * employment_rate)
        model.goods_inventory = monthly_production * 0.5
        
        # Initialize price and wage history
        model.price_history = [model.goods_price]
        model.wage_history = [model.average_wage]
        model.interest_rate_history = [model.interest_rate]
        
        # Store FRED client for updates
        model.real_data_manager = fred_client
        
        logger.info(f"✅ Model initialized successfully:")
        logger.info(f"   Starting Unemployment: {model.unemployment_rate*100:.1f}%")
        logger.info(f"   Starting Inflation: {model.inflation_rate*100:.1f}%")
        logger.info(f"   Starting Interest Rate: {model.interest_rate*100:.1f}%")
        logger.info(f"   Initial Inventory: {model.goods_inventory:.0f} units")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize with FRED data: {e}")
        logger.warning("⚠️  Using default economic parameters")
        
        # Set reasonable defaults
        model.unemployment_rate = 0.04  # 4%
        model.inflation_rate = 0.02  # 2%
        model.interest_rate = 0.02  # 2%
        model.goods_price = 1.0
        model.average_wage = 1.0
        
        # Initialize inventory with defaults
        avg_skill = model.payment_max_skill_multiplier / 2
        monthly_production = (model.n_agents * model.labor_hours * avg_skill * 
                            model.productivity * 0.96)  # 96% employment
        model.goods_inventory = monthly_production * 0.5
        
        # Initialize history
        model.price_history = [model.goods_price]
        model.wage_history = [model.average_wage]
        model.interest_rate_history = [model.interest_rate]
        
        model.real_data_manager = None
        
        return False


def get_fred_calibration_params(fred_api_key: str) -> Dict[str, Any]:
    """
    Get calibration parameters from FRED data.
    
    Args:
        fred_api_key: FRED API key
        
    Returns:
        Dictionary of calibration parameters
    """
    try:
        from ..data_integration.fred_client import FREDClient
        
        fred_client = FREDClient(api_key=fred_api_key)
        snapshot = fred_client.get_current_economic_snapshot()
        
        return {
            'unemployment_target': snapshot.unemployment_rate / 100.0,
            'inflation_target': snapshot.inflation_rate / 100.0,
            'natural_interest_rate': snapshot.fed_funds_rate / 100.0,
            'gdp_growth_target': snapshot.gdp_growth / 100.0,
            'wage_growth_target': snapshot.wage_growth / 100.0
        }
        
    except Exception as e:
        logger.error(f"Failed to get FRED calibration: {e}")
        return {
            'unemployment_target': 0.04,
            'inflation_target': 0.02,
            'natural_interest_rate': 0.02,
            'gdp_growth_target': 0.025,
            'wage_growth_target': 0.03
        }
