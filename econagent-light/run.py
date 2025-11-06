#!/usr/bin/env python3
"""
EconAgent-Light CLI runner.
Main entry point for running economic simulations.
"""

import argparse
import logging
import sys
import os
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import SystemConfig, load_config_from_env
from src.llm_integration import UnifiedLLMClient
from src.mesa_model import EconModel
from src.lightagent_integration import LightAgentWrapper

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from external libraries
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def check_llm_services(config: SystemConfig) -> bool:
    """Check if LLM services are available."""
    logger = logging.getLogger(__name__)
    
    try:
        llm_client = UnifiedLLMClient(
            nemotron_url=config.llm.nemotron_base_url,
            ollama_url=config.llm.ollama_base_url,
            nemotron_model=config.llm.nemotron_model,
            ollama_model=config.llm.ollama_model,
            enable_caching=config.llm.enable_caching,
            cache_size=config.llm.cache_size
        )
        
        health = llm_client.health_check()
        
        if health["any_healthy"]:
            logger.info(f"LLM Services Status: Nemotron={health['nemotron']}, Ollama={health['ollama']}")
            return True
        else:
            logger.warning("No LLM services are available - simulation will use fallback decisions")
            return False
            
    except Exception as e:
        logger.error(f"Failed to check LLM services: {e}")
        return False

def run_simulation(
    n_agents: int = 100,
    years: int = 20,
    seed: Optional[int] = None,
    enable_lightagent: bool = True,
    enable_memory: bool = True,
    enable_tot: bool = True,
    batch_size: int = 8,
    output_dir: str = "./results",
    save_frequency: int = 6,
    log_frequency: int = 3,
    config: Optional[SystemConfig] = None,
    use_real_data: bool = True
) -> EconModel:
    """
    Run economic simulation.
    
    Args:
        n_agents: Number of agents
        years: Number of years to simulate
        seed: Random seed for reproducibility
        enable_lightagent: Enable LightAgent integration
        enable_memory: Enable agent memory
        enable_tot: Enable Tree-of-Thought reasoning
        batch_size: Batch size for LLM requests
        output_dir: Output directory for results
        save_frequency: How often to save results (months)
        log_frequency: How often to log progress (months)
        config: System configuration
        
    Returns:
        Completed EconModel instance
    """
    logger = logging.getLogger(__name__)
    
    if config is None:
        config = load_config_from_env()
    
    # Override config with parameters
    config.economic.n_agents = n_agents
    config.economic.episode_length = years * 12
    config.economic.random_seed = seed
    config.lightagent.enable_memory = enable_memory
    config.lightagent.enable_tot = enable_tot
    config.llm.batch_size = batch_size
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting EconAgent-Light simulation:")
    logger.info(f"  Agents: {n_agents}")
    logger.info(f"  Duration: {years} years ({years * 12} months)")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Real FRED Data: {use_real_data}")
    logger.info(f"  LightAgent: {enable_lightagent}")
    logger.info(f"  Memory: {enable_memory}")
    logger.info(f"  Tree-of-Thought: {enable_tot}")
    
    # Initialize real data if enabled
    real_data_manager = None
    calibrated_params = {}
    
    if use_real_data:
        try:
            from src.data_integration.real_data_manager import RealDataManager
            
            logger.info("Initializing real economic data from FRED...")
            real_data_manager = RealDataManager(
                fred_api_key=config.fred.api_key,
                cache_dir=config.fred.cache_dir,
                auto_update=True
            )
            
            real_data = real_data_manager.initialize_real_data(
                start_date="2010-01-01",
                calibration_scenario="post_covid"
            )
            
            calibrated_params = real_data['calibrated_params']
            logger.info(f"Successfully loaded {len(real_data['data_sources'])} FRED series")
            
        except Exception as e:
            logger.warning(f"Failed to initialize real data: {e}")
            logger.warning("Falling back to default parameters")
            use_real_data = False
    
    # Initialize LLM client
    llm_client = None
    if enable_lightagent:
        try:
            llm_client = UnifiedLLMClient(
                nemotron_url=config.llm.nemotron_base_url,
                ollama_url=config.llm.ollama_base_url,
                nemotron_model=config.llm.nemotron_model,
                ollama_model=config.llm.ollama_model,
                enable_caching=config.llm.enable_caching,
                cache_size=config.llm.cache_size
            )
            
            # Check service health
            health = llm_client.health_check()
            if not health["any_healthy"]:
                logger.warning("No LLM services available - disabling LightAgent")
                llm_client = None
                enable_lightagent = False
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            logger.warning("Disabling LightAgent due to LLM client failure")
            llm_client = None
            enable_lightagent = False
    
    # Initialize model
    logger.info("Initializing economic model...")
    
    # Use real data parameters if available, otherwise use config defaults
    if use_real_data and calibrated_params:
        economic_params = {
            'productivity': calibrated_params.get('productivity', config.economic.productivity),
            'max_price_inflation': calibrated_params.get('max_price_inflation', config.economic.max_price_inflation),
            'max_wage_inflation': calibrated_params.get('max_wage_inflation', config.economic.max_wage_inflation),
            'base_interest_rate': calibrated_params.get('base_interest_rate', config.economic.base_interest_rate),
            'pareto_param': calibrated_params.get('pareto_param', config.economic.pareto_param),
            'payment_max_skill_multiplier': calibrated_params.get('payment_max_skill_multiplier', config.economic.payment_max_skill_multiplier),
        }
        logger.info("Using FRED-calibrated economic parameters")
    else:
        economic_params = {
            'productivity': config.economic.productivity,
            'max_price_inflation': config.economic.max_price_inflation,
            'max_wage_inflation': config.economic.max_wage_inflation,
            'base_interest_rate': config.economic.base_interest_rate,
            'pareto_param': config.economic.pareto_param,
            'payment_max_skill_multiplier': config.economic.payment_max_skill_multiplier,
        }
        logger.info("Using default economic parameters")
    
    model = EconModel(
        n_agents=n_agents,
        episode_length=years * 12,
        random_seed=seed,
        # Economic parameters (real or default)
        productivity=economic_params['productivity'],
        skill_change=config.economic.skill_change,
        price_change=config.economic.price_change,
        max_price_inflation=economic_params['max_price_inflation'],
        max_wage_inflation=economic_params['max_wage_inflation'],
        pareto_param=economic_params['pareto_param'],
        payment_max_skill_multiplier=economic_params['payment_max_skill_multiplier'],
        labor_hours=config.economic.labor_hours,
        consumption_rate_step=config.economic.consumption_rate_step,
        base_interest_rate=economic_params['base_interest_rate'],
        # Real FRED data integration (full ACL24-EconAgent paper implementation)
        fred_api_key=config.fred.api_key,
        enable_real_data=use_real_data,
        real_data_update_frequency=12,  # Update real data annually
        # LLM and LightAgent
        llm_client=llm_client,
        enable_lightagent=enable_lightagent,
        enable_memory=enable_memory,
        enable_tot=enable_tot,
        # Simulation parameters
        save_frequency=save_frequency,
        log_frequency=log_frequency
    )
    
    # Attach real data manager for validation
    if real_data_manager:
        model.real_data_manager = real_data_manager
    
    # Run simulation
    logger.info("Starting simulation...")
    start_time = time.time()
    
    try:
        step_count = 0
        while model.running and step_count < years * 12:
            model.step()
            step_count += 1
            
            # Save intermediate results
            if step_count % save_frequency == 0:
                output_file = os.path.join(output_dir, f"results_step_{step_count}.xlsx")
                model.save_results(output_file)
                logger.info(f"Saved intermediate results to {output_file}")
        
        # Final save
        final_output = os.path.join(output_dir, "final_results.xlsx")
        model.save_results(final_output)
        
        # Save summary statistics
        summary = model.get_summary_stats()
        summary_file = os.path.join(output_dir, "summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("EconAgent-Light Simulation Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Real FRED Data Used: {use_real_data}\n")
            f.write(f"Simulation Length: {summary['simulation_length']} months\n")
            f.write(f"Final GDP: ${summary['final_gdp']:,.2f}\n")
            f.write(f"Average Unemployment: {summary['avg_unemployment']:.1%}\n")
            f.write(f"Average Inflation: {summary['avg_inflation']:.1%}\n")
            f.write(f"Final Gini Coefficient: {summary['final_gini']:.3f}\n")
            f.write(f"Total Agents: {summary['total_agents']}\n\n")
            
            if summary['llm_stats']:
                f.write("LLM Statistics:\n")
                for key, value in summary['llm_stats'].items():
                    f.write(f"  {key}: {value}\n")
        
        # Generate real data report if available
        if real_data_manager:
            real_data_report = real_data_manager.generate_data_report(
                os.path.join(output_dir, "real_data_report.txt")
            )
            logger.info("Real data integration report saved")
            
            # Validate simulation results against real data
            try:
                simulation_results = model.get_results_dataframe()
                validation_scores = real_data_manager.validate_simulation_results(simulation_results)
                
                validation_file = os.path.join(output_dir, "validation_scores.txt")
                with open(validation_file, 'w') as f:
                    f.write("Simulation Validation Against Real FRED Data\n")
                    f.write("=" * 50 + "\n\n")
                    for metric, scores in validation_scores.items():
                        if isinstance(scores, dict):
                            f.write(f"{metric.upper()}:\n")
                            for score_type, value in scores.items():
                                f.write(f"  {score_type}: {value:.4f}\n")
                        else:
                            f.write(f"{metric}: {scores:.4f}\n")
                        f.write("\n")
                
                logger.info("Validation scores saved")
                
            except Exception as e:
                logger.warning(f"Failed to validate simulation results: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Simulation completed in {elapsed_time:.1f} seconds")
        logger.info(f"Results saved to {output_dir}")
        
        return model
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        return model
    
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="EconAgent-Light: Economic simulation with LightAgent + local LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Simulation parameters
    parser.add_argument("--agents", type=int, default=100,
                       help="Number of economic agents")
    parser.add_argument("--years", type=int, default=20,
                       help="Number of years to simulate")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    # Data parameters
    parser.add_argument("--no-real-data", action="store_true",
                       help="Disable real FRED data integration (use default parameters)")
    
    # LightAgent parameters
    parser.add_argument("--no-lightagent", action="store_true",
                       help="Disable LightAgent integration")
    parser.add_argument("--no-memory", action="store_true",
                       help="Disable agent memory")
    parser.add_argument("--no-tot", action="store_true",
                       help="Disable Tree-of-Thought reasoning")
    
    # LLM parameters
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for LLM requests")
    parser.add_argument("--nemotron-url", type=str, default="http://localhost:8000/v1",
                       help="Nemotron service URL")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434/v1",
                       help="Ollama service URL")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--save-frequency", type=int, default=6,
                       help="Save results every N months")
    
    # Logging parameters
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Log file path (optional)")
    parser.add_argument("--log-frequency", type=int, default=3,
                       help="Log progress every N months")
    
    # Quick test mode
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick test (10 agents, 1 year)")
    
    # Service check
    parser.add_argument("--check-services", action="store_true",
                       help="Check LLM service availability and exit")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config_from_env()
    
    # Override config with CLI arguments
    if args.nemotron_url:
        config.llm.nemotron_base_url = args.nemotron_url
    if args.ollama_url:
        config.llm.ollama_base_url = args.ollama_url
    
    # Service check mode
    if args.check_services:
        logger.info("Checking LLM service availability...")
        available = check_llm_services(config)
        if available:
            logger.info("✓ LLM services are available")
            sys.exit(0)
        else:
            logger.error("✗ No LLM services are available")
            sys.exit(1)
    
    # Quick test mode
    if args.quick_test:
        logger.info("Running in quick test mode")
        args.agents = 10
        args.years = 1
        args.log_frequency = 1
        args.save_frequency = 3
    
    # Validate parameters
    if args.agents < 1:
        logger.error("Number of agents must be at least 1")
        sys.exit(1)
    
    if args.years < 1:
        logger.error("Number of years must be at least 1")
        sys.exit(1)
    
    # Run simulation
    try:
        model = run_simulation(
            n_agents=args.agents,
            years=args.years,
            seed=args.seed,
            enable_lightagent=not args.no_lightagent,
            enable_memory=not args.no_memory,
            enable_tot=not args.no_tot,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            save_frequency=args.save_frequency,
            log_frequency=args.log_frequency,
            config=config,
            use_real_data=not args.no_real_data
        )
        
        logger.info("Simulation completed successfully!")
        
        # Print summary
        summary = model.get_summary_stats()
        print("\n" + "="*50)
        print("SIMULATION SUMMARY")
        print("="*50)
        print(f"Duration: {summary['simulation_length']} months")
        print(f"Final GDP: ${summary['final_gdp']:,.2f}")
        print(f"Avg Unemployment: {summary['avg_unemployment']:.1%}")
        print(f"Avg Inflation: {summary['avg_inflation']:.1%}")
        print(f"Final Inequality (Gini): {summary['final_gini']:.3f}")
        print(f"Results saved to: {args.output_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()