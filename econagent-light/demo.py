#!/usr/bin/env python3
"""
EconAgent-Light Demo Script
Demonstrates the system with a small simulation.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import SystemConfig
from src.mesa_model import EconModel
from src.viz import create_analysis_report

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def run_demo():
    """Run a small demonstration simulation."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting EconAgent-Light Demo")
    logger.info("=" * 50)
    
    # Create a small simulation (no LLM required)
    logger.info("Initializing model with 10 agents for 12 months...")
    
    model = EconModel(
        n_agents=10,
        episode_length=12,  # 1 year
        random_seed=42,
        llm_client=None,  # No LLM for demo
        enable_lightagent=False,
        log_frequency=3
    )
    
    logger.info("Running simulation...")
    
    # Run simulation
    step_count = 0
    while model.running and step_count < 12:
        model.step()
        step_count += 1
    
    logger.info("Simulation completed!")
    
    # Get results
    results_df = model.get_results_dataframe()
    summary = model.get_summary_stats()
    
    # Print summary
    print("\n" + "="*50)
    print("DEMO SIMULATION RESULTS")
    print("="*50)
    print(f"Simulation Length: {summary['simulation_length']} months")
    print(f"Final GDP: ${summary['final_gdp']:,.2f}")
    print(f"Average Unemployment: {summary['avg_unemployment']:.1%}")
    print(f"Average Inflation: {summary['avg_inflation']:.1%}")
    print(f"Final Gini Coefficient: {summary['final_gini']:.3f}")
    print(f"Total Agents: {summary['total_agents']}")
    
    # Show some economic indicators
    print("\nEconomic Indicators by Month:")
    print("-" * 30)
    for i, row in results_df.iterrows():
        print(f"Month {row['Month']:2d}: GDP=${row['GDP']:6.0f}, "
              f"Unemployment={row['Unemployment']:5.1%}, "
              f"Inflation={row['Inflation']:6.1%}")
    
    # Save results
    output_dir = Path("./demo_results")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "demo_results.xlsx"
    model.save_results(str(results_file))
    
    logger.info(f"Results saved to {results_file}")
    
    # Create analysis (if matplotlib available)
    try:
        create_analysis_report(str(results_file), str(output_dir / "analysis"))
        logger.info(f"Analysis report created in {output_dir / 'analysis'}")
    except ImportError:
        logger.warning("Matplotlib not available - skipping plots")
    except Exception as e:
        logger.warning(f"Analysis failed: {e}")
    
    print("\n" + "="*50)
    print("Demo completed successfully!")
    print(f"Check {output_dir} for detailed results")
    print("="*50)

if __name__ == "__main__":
    setup_logging()
    
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        sys.exit(1)