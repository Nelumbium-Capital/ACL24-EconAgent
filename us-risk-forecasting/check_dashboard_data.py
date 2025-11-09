"""
Check what data the dashboard is actually displaying.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
os.environ['FRED_API_KEY'] = 'bcc1a43947af1745a35bfb3b7132b7c6'

# Import after setting env
from src.dashboard.app import data_cache, fetch_and_process_data
import numpy as np

print("="*70)
print("CHECKING DASHBOARD DATA")
print("="*70)

# Trigger data fetch
print("\nFetching dashboard data...")
fetch_and_process_data()

print("\n1. ECONOMIC DATA:")
if data_cache['economic_data'] is not None:
    df = data_cache['economic_data']
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Last 3 observations:")
    print(df.tail(3))
else:
    print("   ✗ No economic data")

print("\n2. FORECASTS:")
if data_cache['forecasts'] is not None:
    df = data_cache['forecasts']
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   First 3 forecasts:")
    print(df.head(3))
    print(f"\n   Last 3 forecasts:")
    print(df.tail(3))
    
    # Check if forecasts are varying
    for col in df.columns:
        values = df[col].values
        std = values.std()
        min_val = values.min()
        max_val = values.max()
        print(f"\n   {col}:")
        print(f"     Range: {min_val:.4f} to {max_val:.4f}")
        print(f"     Std dev: {std:.4f}")
        if std < 0.001:
            print(f"     ⚠ WARNING: Very low variation (flat forecast)")
        else:
            print(f"     ✓ Showing variation")
else:
    print("   ✗ No forecasts")

print("\n3. MODEL FORECASTS:")
if data_cache['model_forecasts'] is not None:
    mf = data_cache['model_forecasts']
    print(f"   Indicators: {list(mf.keys())}")
    
    for indicator, models in mf.items():
        print(f"\n   {indicator}:")
        for model_name, forecast in models.items():
            if isinstance(forecast, np.ndarray):
                print(f"     {model_name}: {forecast[0]:.4f} to {forecast[-1]:.4f} (std={forecast.std():.4f})")
            else:
                print(f"     {model_name}: {type(forecast)}")
else:
    print("   ✗ No model forecasts")

print("\n4. SCENARIO RESULTS:")
if data_cache['scenario_results'] is not None:
    sr = data_cache['scenario_results']
    print(f"   Scenarios: {list(sr.keys())}")
    
    for scenario_name, result in sr.items():
        if result is not None:
            default_rate = result['simulation']['default_rate'].mean() * 100
            print(f"   {scenario_name}: {default_rate:.2f}% mean default rate")
        else:
            print(f"   {scenario_name}: Failed")
else:
    print("   ✗ No scenario results")

print("\n5. KRIs:")
if data_cache['kris'] is not None:
    kris = data_cache['kris']
    print(f"   Total KRIs: {len(kris)}")
    for kri_name, value in list(kris.items())[:5]:
        print(f"     {kri_name}: {value:.2f}")
else:
    print("   ✗ No KRIs")

print("\n" + "="*70)
print("DASHBOARD DATA CHECK COMPLETE")
print("="*70)
