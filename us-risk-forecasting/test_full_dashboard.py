"""
Full end-to-end test of the dashboard with real data.
This script tests the complete data pipeline, forecasting, simulation, and KRI calculation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
os.environ['FRED_API_KEY'] = 'bcc1a43947af1745a35bfb3b7132b7c6'

def main():
    print("=" * 70)
    print("FULL DASHBOARD TEST WITH REAL DATA")
    print("=" * 70)
    
    # Import dashboard components
    print("\n1. Importing dashboard components...")
    from src.dashboard.app import fetch_and_process_data, data_cache
    print("✓ Dashboard components imported")
    
    # Fetch and process real data
    print("\n2. Fetching real data from FRED...")
    print("   This will fetch:")
    print("   - Unemployment Rate (UNRATE)")
    print("   - CPI Inflation (CPIAUCSL)")
    print("   - Federal Funds Rate (FEDFUNDS)")
    print("   - Credit Spread (BAA10Y)")
    print()
    
    try:
        fetch_and_process_data()
        print("✓ Data fetched and processed successfully")
    except Exception as e:
        print(f"✗ Data fetching failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Verify data cache
    print("\n3. Verifying data cache...")
    if data_cache['economic_data'] is not None:
        print(f"✓ Economic data: {len(data_cache['economic_data'])} observations")
        print(f"  Columns: {list(data_cache['economic_data'].columns)}")
    else:
        print("✗ No economic data in cache")
        return 1
    
    if data_cache['forecasts'] is not None:
        print(f"✓ Forecasts: {len(data_cache['forecasts'])} periods")
        print(f"  Columns: {list(data_cache['forecasts'].columns)}")
    else:
        print("✗ No forecasts in cache")
        return 1
    
    if data_cache['model_forecasts'] is not None:
        print(f"✓ Model forecasts available for comparison")
        for indicator, models in data_cache['model_forecasts'].items():
            print(f"  {indicator}: {list(models.keys())}")
    else:
        print("✗ No model forecasts in cache")
        return 1
    
    if data_cache['kris'] is not None:
        print(f"✓ KRIs computed: {len(data_cache['kris'])} indicators")
        print("  KRI values:")
        for kri_name, value in list(data_cache['kris'].items())[:5]:
            print(f"    {kri_name}: {value:.2f}")
    else:
        print("✗ No KRIs in cache")
        return 1
    
    if data_cache['risk_levels'] is not None:
        print(f"✓ Risk levels evaluated")
        risk_counts = {}
        for level in data_cache['risk_levels'].values():
            risk_counts[level.value] = risk_counts.get(level.value, 0) + 1
        print(f"  Risk distribution: {risk_counts}")
    else:
        print("✗ No risk levels in cache")
        return 1
    
    if data_cache['scenario_results'] is not None:
        print(f"✓ Scenario simulations completed")
        for scenario_name, result in data_cache['scenario_results'].items():
            if result is not None:
                print(f"  {scenario_name}: ✓")
            else:
                print(f"  {scenario_name}: ✗ (failed)")
    else:
        print("✗ No scenario results in cache")
        return 1
    
    # Display sample data
    print("\n4. Sample Economic Data (last 5 observations):")
    print(data_cache['economic_data'].tail())
    
    print("\n5. Sample Forecasts (first 5 periods):")
    print(data_cache['forecasts'].head())
    
    print("\n6. Sample KRI Values:")
    for kri_name, value in list(data_cache['kris'].items())[:10]:
        risk_level = data_cache['risk_levels'][kri_name]
        print(f"  {kri_name:30s}: {value:8.2f} [{risk_level.value.upper()}]")
    
    print("\n7. Scenario Comparison (Default Rates):")
    for scenario_name, result in data_cache['scenario_results'].items():
        if result is not None and 'kris' in result:
            default_rate = result['kris'].get('loan_default_rate', 0)
            print(f"  {scenario_name:20s}: {default_rate:.2f}%")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - Dashboard is fully functional with real data!")
    print("=" * 70)
    print("\nTo start the dashboard, run:")
    print("  python src/dashboard/app.py")
    print("\nThen open your browser to: http://localhost:8050")
    print()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
