"""
Test script for the Mesa-based stress testing simulation.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from simulation import (
    RiskSimulationModel,
    RecessionScenario,
    InterestRateShockScenario,
    CreditCrisisScenario,
    MonteCarloEngine,
    run_stress_test
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


def test_monte_carlo():
    """Test Monte Carlo simulation."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Monte Carlo simulation")
    logger.info("=" * 60)
    
    scenario = RecessionScenario()
    
    logger.info("Running 10 Monte Carlo simulations...")
    summary_stats, risk_metrics = run_stress_test(
        scenario=scenario,
        n_simulations=10,
        n_banks=5,
        n_firms=20,
        n_steps=50,
        random_seed=42
    )
    
    logger.info("\nRisk Metrics for Default Rate:")
    dr_metrics = risk_metrics['default_rate']
    logger.info(f"  Mean: {dr_metrics['mean']:.4f}")
    logger.info(f"  Median: {dr_metrics['median']:.4f}")
    logger.info(f"  Std Dev: {dr_metrics['std']:.4f}")
    logger.info(f"  VaR (95%): {dr_metrics['var_95']:.4f}")
    logger.info(f"  CVaR (95%): {dr_metrics['cvar_95']:.4f}")
    
    return summary_stats, risk_metrics


def test_scenario_comparison():
    """Test scenario comparison."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing scenario comparison")
    logger.info("=" * 60)
    
    engine = MonteCarloEngine(n_banks=5, n_firms=20, n_steps=50)
    
    scenarios = [
        RecessionScenario(),
        InterestRateShockScenario(),
        CreditCrisisScenario()
    ]
    
    logger.info("Comparing scenarios with 5 simulations each...")
    comparison = engine.compare_scenarios(
        scenarios=scenarios,
        n_simulations=5,
        random_seed=42
    )
    
    # Show default rate comparison
    default_rate_comparison = comparison[comparison['metric'] == 'default_rate']
    logger.info("\nDefault Rate Comparison:")
    for _, row in default_rate_comparison.iterrows():
        logger.info(f"  {row['scenario']}: mean={row['mean']:.4f}, "
                   f"var_95={row['var_95']:.4f}")
    
    return comparison


if __name__ == '__main__':
    try:
        # Run all tests
        test_single_simulation()
        test_multiple_scenarios()
        test_monte_carlo()
        test_scenario_comparison()
        
        logger.info("\n" + "=" * 60)
        logger.info("All tests completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
