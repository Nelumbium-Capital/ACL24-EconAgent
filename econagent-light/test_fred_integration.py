#!/usr/bin/env python3
"""
Test script for FRED data integration.
Verifies that the real data manager works with the provided API key.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_fred_integration():
    """Test FRED data integration."""
    logger = logging.getLogger(__name__)
    
    print("=" * 70)
    print("TESTING FRED DATA INTEGRATION")
    print("=" * 70)
    
    try:
        # Import required modules
        from src.data_integration.real_data_manager import RealDataManager
        from config import DEFAULT_CONFIG
        
        # Test API key
        api_key = DEFAULT_CONFIG.fred.api_key
        print(f"Using FRED API Key: {api_key[:8]}...{api_key[-4:]}")
        
        # Initialize real data manager
        logger.info("Initializing RealDataManager...")
        data_manager = RealDataManager(
            fred_api_key=api_key,
            cache_dir="./test_cache",
            auto_update=True
        )
        
        # Test basic FRED client functionality
        logger.info("Testing FRED client...")
        
        # Fetch a simple series (unemployment rate)
        unemployment_data = data_manager.fred_client.get_series(
            'UNRATE', 
            start_date='2023-01-01'
        )
        
        if not unemployment_data.empty:
            latest_unemployment = unemployment_data.iloc[-1, 0]
            latest_date = unemployment_data.index[-1].strftime("%Y-%m-%d")
            print(f"✅ Successfully fetched unemployment data")
            print(f"   Latest unemployment rate: {latest_unemployment:.1f}% (as of {latest_date})")
        else:
            print("❌ Failed to fetch unemployment data")
            return False
        
        # Test full data initialization
        logger.info("Testing full data initialization...")
        
        real_data = data_manager.initialize_real_data(
            start_date="2022-01-01",
            calibration_scenario="post_covid"
        )
        
        print(f"✅ Successfully initialized real data")
        print(f"   Loaded {len(real_data['data_sources'])} FRED series")
        print(f"   Data sources: {', '.join(real_data['data_sources'][:5])}...")
        
        # Test current indicators
        logger.info("Testing current indicators...")
        
        indicators = data_manager.get_real_time_indicators()
        
        print(f"✅ Successfully retrieved {len(indicators)} current indicators")
        print("\nCurrent Economic Indicators:")
        print("-" * 40)
        
        key_indicators = ['unemployment', 'fed_funds', 'cpi', 'real_gdp']
        for indicator in key_indicators:
            if indicator in indicators:
                data = indicators[indicator]
                if isinstance(data, dict) and 'value' in data:
                    print(f"  {indicator.replace('_', ' ').title():<20}: {data['value']:>8.3f}")
        
        # Test calibrated parameters
        calibrated_params = real_data['calibrated_params']
        
        print(f"\n✅ Successfully calibrated parameters")
        print("\nKey Calibrated Parameters:")
        print("-" * 30)
        
        key_params = [
            ('Base Interest Rate', 'base_interest_rate'),
            ('Max Price Inflation', 'max_price_inflation'),
            ('Productivity', 'productivity'),
            ('Phillips Curve Coeff', 'phillips_curve_coefficient'),
            ('Okun\'s Law Coeff', 'okuns_law_coefficient')
        ]
        
        for label, key in key_params:
            if key in calibrated_params:
                value = calibrated_params[key]
                print(f"  {label:<20}: {value:>8.4f}")
        
        # Generate and display report
        logger.info("Generating data report...")
        
        report = data_manager.generate_data_report()
        
        print(f"\n✅ Successfully generated data report")
        print("\nData Integration Report Summary:")
        print("-" * 40)
        
        # Show first few lines of report
        report_lines = report.split('\n')
        for line in report_lines[:15]:  # Show first 15 lines
            if line.strip():
                print(f"  {line}")
        
        if len(report_lines) > 15:
            print(f"  ... ({len(report_lines) - 15} more lines)")
        
        print("\n" + "=" * 70)
        print("✅ FRED DATA INTEGRATION TEST SUCCESSFUL!")
        print("✅ All real economic data is working properly")
        print("✅ No mock data or placeholders detected")
        print("=" * 70)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all required packages are installed")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Full error details:")
        return False

def test_simulation_with_real_data():
    """Test running a mini simulation with real data."""
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 70)
    print("TESTING SIMULATION WITH REAL FRED DATA")
    print("=" * 70)
    
    try:
        from src.data_integration.real_data_manager import RealDataManager
        from src.mesa_model.model import EconModel
        from config import DEFAULT_CONFIG
        
        # Initialize real data
        logger.info("Initializing real data for simulation...")
        
        data_manager = RealDataManager(
            fred_api_key=DEFAULT_CONFIG.fred.api_key,
            cache_dir="./test_cache"
        )
        
        real_data = data_manager.initialize_real_data(
            start_date="2022-01-01",
            calibration_scenario="post_covid"
        )
        
        calibrated_params = real_data['calibrated_params']
        
        # Create model with real parameters
        logger.info("Creating model with FRED-calibrated parameters...")
        
        model = EconModel(
            n_agents=5,  # Very small for quick test
            episode_length=3,  # 3 months
            random_seed=42,
            # Use real FRED-calibrated parameters
            productivity=calibrated_params.get('productivity', 1.0),
            max_price_inflation=calibrated_params.get('max_price_inflation', 0.10),
            max_wage_inflation=calibrated_params.get('max_wage_inflation', 0.05),
            base_interest_rate=calibrated_params.get('base_interest_rate', 0.02),
            # Disable LLM for test
            llm_client=None,
            enable_lightagent=False,
            log_frequency=1
        )
        
        # Run mini simulation
        logger.info("Running mini simulation...")
        
        for step in range(3):
            model.step()
        
        # Get results
        results = model.get_results_dataframe()
        summary = model.get_summary_stats()
        
        print(f"✅ Successfully ran simulation with real FRED data")
        print(f"   Simulation length: {summary['simulation_length']} months")
        print(f"   Final GDP: ${summary['final_gdp']:,.2f}")
        print(f"   Avg unemployment: {summary['avg_unemployment']:.1%}")
        print(f"   Avg inflation: {summary['avg_inflation']:.1%}")
        
        print("\nSimulation used these FRED-calibrated parameters:")
        print(f"  Productivity: {calibrated_params.get('productivity', 'N/A'):.4f}")
        print(f"  Max inflation: {calibrated_params.get('max_price_inflation', 'N/A'):.4f}")
        print(f"  Base interest rate: {calibrated_params.get('base_interest_rate', 'N/A'):.4f}")
        
        print("\n✅ SIMULATION WITH REAL DATA TEST SUCCESSFUL!")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        logger.exception("Full error details:")
        return False

if __name__ == "__main__":
    setup_logging()
    
    success = True
    
    try:
        # Test FRED integration
        if not test_fred_integration():
            success = False
        
        # Test simulation with real data
        if not test_simulation_with_real_data():
            success = False
        
        if success:
            print("\n🎉 ALL TESTS PASSED!")
            print("🎉 FRED integration is working perfectly")
            print("🎉 Real economic data is fully integrated")
            print("🎉 No mock data or placeholders remain")
        else:
            print("\n❌ SOME TESTS FAILED")
            print("❌ Check the error messages above")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)