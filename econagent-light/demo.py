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
    
    logger.info("Starting EconAgent-Light Demo with Real FRED Data")
    logger.info("=" * 60)
    
    # Initialize real data integration
    try:
        from src.data_integration.real_data_manager import RealDataManager
        from config import DEFAULT_CONFIG
        
        logger.info("Initializing real economic data from FRED...")
        data_manager = RealDataManager(
            fred_api_key=DEFAULT_CONFIG.fred.api_key,
            cache_dir=DEFAULT_CONFIG.fred.cache_dir,
            auto_update=True
        )
        
        # Get real economic data
        real_data = data_manager.initialize_real_data(
            start_date="2020-01-01",
            calibration_scenario="post_covid"
        )
        
        calibrated_params = real_data['calibrated_params']
        logger.info(f"Successfully loaded {len(real_data['data_sources'])} FRED series")
        
        # Show current economic indicators
        current_indicators = data_manager.get_real_time_indicators()
        logger.info("Current Economic Conditions:")
        for name, data in list(current_indicators.items())[:5]:
            if isinstance(data, dict) and 'value' in data:
                logger.info(f"  {name}: {data['value']:.3f}")
        
        use_real_data = True
        
    except Exception as e:
        logger.warning(f"Failed to initialize real data: {e}")
        logger.warning("Running demo with default parameters")
        calibrated_params = {}
        use_real_data = False
    
    # Create a small simulation
    logger.info("Initializing model with 10 agents for 12 months...")
    
    if use_real_data:
        model = EconModel(
            n_agents=10,
            episode_length=12,  # 1 year
            random_seed=42,
            # Use real FRED-calibrated parameters
            productivity=calibrated_params.get('productivity', 1.0),
            max_price_inflation=calibrated_params.get('max_price_inflation', 0.10),
            max_wage_inflation=calibrated_params.get('max_wage_inflation', 0.05),
            base_interest_rate=calibrated_params.get('base_interest_rate', 0.02),
            llm_client=None,  # No LLM for demo
            enable_lightagent=False,
            log_frequency=3
        )
        model.real_data_manager = data_manager
    else:
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
    print("\n" + "="*60)
    print("DEMO SIMULATION RESULTS")
    print("="*60)
    print(f"Real FRED Data Used: {use_real_data}")
    print(f"Simulation Length: {summary['simulation_length']} months")
    print(f"Final GDP: ${summary['final_gdp']:,.2f}")
    print(f"Average Unemployment: {summary['avg_unemployment']:.1%}")
    print(f"Average Inflation: {summary['avg_inflation']:.1%}")
    print(f"Final Gini Coefficient: {summary['final_gini']:.3f}")
    print(f"Total Agents: {summary['total_agents']}")
    
    if use_real_data and calibrated_params:
        print("\nFRED-Calibrated Parameters Used:")
        print(f"  Productivity: {calibrated_params.get('productivity', 'N/A'):.4f}")
        print(f"  Max Inflation: {calibrated_params.get('max_price_inflation', 'N/A'):.4f}")
        print(f"  Base Interest Rate: {calibrated_params.get('base_interest_rate', 'N/A'):.4f}")
        if 'phillips_curve_coefficient' in calibrated_params:
            print(f"  Phillips Curve Coeff: {calibrated_params['phillips_curve_coefficient']:.4f}")
        if 'okuns_law_coefficient' in calibrated_params:
            print(f"  Okun's Law Coeff: {calibrated_params['okuns_law_coefficient']:.4f}")
    
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
    
    # Generate real data report if available
    if use_real_data and hasattr(model, 'real_data_manager'):
        real_data_report = model.real_data_manager.generate_data_report(
            str(output_dir / "real_data_report.txt")
        )
        logger.info("Real data integration report saved")
    
    # Create analysis (if matplotlib available)
    try:
        create_analysis_report(str(results_file), str(output_dir / "analysis"))
        logger.info(f"Analysis report created in {output_dir / 'analysis'}")
    except ImportError:
        logger.warning("Matplotlib not available - skipping plots")
    except Exception as e:
        logger.warning(f"Analysis failed: {e}")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print(f"Check {output_dir} for detailed results")
    if use_real_data:
        print("Real FRED economic data was successfully integrated!")
    print("="*60)

if __name__ == "__main__":
    setup_logging()
    
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        sys.exit(1)