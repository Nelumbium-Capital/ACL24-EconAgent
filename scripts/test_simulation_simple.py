"""
Simple test script for the Mesa-based stress testing simulation.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from simulation import (
    RiskSimulationModel,
    RecessionScenario,
    InterestRateShockScenario,
    CreditCrisisScenario
)
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_simulation():
    """Test a single simulation run."""
    logger.info("=" * 60)
    logger.info("Testing single simulation with recession scenario")
    logger.info("=" * 60)
    
    scenario = RecessionScenario(shock_start=10, shock_duration=20)
    model = RiskSimulationModel(
        n_banks=5,
        n_firms=20,
        scenario=scenario,
        random_seed=42
    )
    
    results = model.run_simulation(n_steps=50)
    
    logger.info(f"\nSimulation Results:")
    logger.info(f"  Final default rate: {results['default_rate'].iloc[-1]:.4f}")
    logger.info(f"  Final system liquidity: {results['system_liquidity'].iloc[-1]:.4f}")
    logger.info(f"  Final network stress: {results['network_stress'].iloc[-1]:.4f}")
    logger.info(f"  Final avg capital ratio: {results['avg_capital_ratio'].iloc[-1]:.4f}")
    
    return results


def test_multiple_scenarios():
    """Test different scenarios."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing multiple scenarios")
    logger.info("=" * 60)
    
    scenarios = [
        RecessionScenario(),
        InterestRateShockScenario(),
        CreditCrisisScenario()
    ]
    
    for scenario in scenarios:
        logger.info(f"\nRunning {scenario.name} scenario...")
        model = RiskSimulationModel(
            n_banks=5,
            n_firms=20,
            scenario=scenario,
            random_seed=42
        )
        
        results = model.run_simulation(n_steps=50)
        
        logger.info(f"  Peak default rate: {results['default_rate'].max():.4f}")
        logger.info(f"  Min system liquidity: {results['system_liquidity'].min():.4f}")


if __name__ == '__main__':
    try:
        # Run tests
        test_single_simulation()
        test_multiple_scenarios()
        
        logger.info("\n" + "=" * 60)
        logger.info("All tests completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
