#!/usr/bin/env python3
"""
Test script to run the Mesa economic model directly and verify it generates realistic data.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.mesa_model.model import EconModel
from dotenv import load_dotenv
import pandas as pd

# Load environment
load_dotenv()

def test_mesa_model():
    """Test the Mesa economic model with realistic parameters."""
    
    print("=" * 80)
    print("Testing Mesa EconModel")
    print("=" * 80)
    
    # Create model with realistic parameters
    model = EconModel(
        n_agents=100,
        episode_length=60,  # 5 years for quick test
        random_seed=42,
        productivity=1.0,
        skill_change=0.02,
        price_change=0.02,
        max_price_inflation=0.05,  # 5% max
        max_wage_inflation=0.03,   # 3% max
        base_interest_rate=0.02,   # 2%
        fred_api_key=os.getenv("FRED_API_KEY"),
        enable_real_data=True,
        real_data_update_frequency=12,
        save_frequency=6,
        log_frequency=12
    )
    
    print(f"\nModel initialized:")
    print(f"  Agents: {model.n_agents}")
    print(f"  Steps: {model.episode_length}")
    print(f"  Initial unemployment: {model.unemployment_rate*100:.1f}%")
    print(f"  Initial inflation: {model.inflation_rate*100:.1f}%")
    print(f"  Initial price: ${model.goods_price:.2f}")
    print(f"  Initial wage: ${model.average_wage:.2f}")
    print(f"  FRED data enabled: {model.enable_real_data}")
    
    # Run simulation
    print("\nRunning simulation...")
    for step in range(model.episode_length):
        model.step()
        
        if (step + 1) % 12 == 0:
            year = (step + 1) // 12
            print(f"  Year {year}: "
                  f"Unemployment={model.unemployment_rate*100:.1f}%, "
                  f"Inflation={model.inflation_rate*100:.1f}%, "
                  f"Price=${model.goods_price:.2f}, "
                  f"Wage=${model.average_wage:.2f}")
    
    # Get results
    results_df = model.get_results_dataframe()
    
    print("\n" + "=" * 80)
    print("Simulation Results Summary")
    print("=" * 80)
    
    # Calculate statistics
    unemployment_rates = results_df['Unemployment'] * 100
    inflation_rates = results_df['Inflation'] * 100
    gdp_values = results_df['GDP']
    
    # Calculate GDP growth
    gdp_growth = []
    for i in range(1, len(gdp_values)):
        if gdp_values.iloc[i-1] > 0:
            growth = ((gdp_values.iloc[i] - gdp_values.iloc[i-1]) / gdp_values.iloc[i-1]) * 100
            gdp_growth.append(growth)
    
    print(f"\nUnemployment Rate:")
    print(f"  Mean: {unemployment_rates.mean():.2f}%")
    print(f"  Std:  {unemployment_rates.std():.2f}%")
    print(f"  Min:  {unemployment_rates.min():.2f}%")
    print(f"  Max:  {unemployment_rates.max():.2f}%")
    print(f"  First 10: {unemployment_rates.head(10).tolist()}")
    
    print(f"\nInflation Rate:")
    print(f"  Mean: {inflation_rates.mean():.2f}%")
    print(f"  Std:  {inflation_rates.std():.2f}%")
    print(f"  Min:  {inflation_rates.min():.2f}%")
    print(f"  Max:  {inflation_rates.max():.2f}%")
    print(f"  First 10: {inflation_rates.head(10).tolist()}")
    
    if gdp_growth:
        print(f"\nGDP Growth:")
        print(f"  Mean: {sum(gdp_growth)/len(gdp_growth):.2f}%")
        print(f"  Std:  {pd.Series(gdp_growth).std():.2f}%")
        print(f"  Min:  {min(gdp_growth):.2f}%")
        print(f"  Max:  {max(gdp_growth):.2f}%")
        print(f"  First 10: {gdp_growth[:10]}")
    
    # Check for issues
    print("\n" + "=" * 80)
    print("Data Quality Check")
    print("=" * 80)
    
    issues = []
    
    if unemployment_rates.mean() < 1 or unemployment_rates.mean() > 15:
        issues.append(f"❌ Unrealistic unemployment mean: {unemployment_rates.mean():.1f}%")
    else:
        print(f"✅ Unemployment mean is realistic: {unemployment_rates.mean():.1f}%")
    
    if inflation_rates.mean() < -2 or inflation_rates.mean() > 10:
        issues.append(f"❌ Unrealistic inflation mean: {inflation_rates.mean():.1f}%")
    else:
        print(f"✅ Inflation mean is realistic: {inflation_rates.mean():.1f}%")
    
    if unemployment_rates.std() > 5:
        issues.append(f"❌ Unemployment too volatile: std={unemployment_rates.std():.1f}%")
    else:
        print(f"✅ Unemployment volatility is reasonable: {unemployment_rates.std():.1f}%")
    
    if inflation_rates.std() > 3:
        issues.append(f"❌ Inflation too volatile: std={inflation_rates.std():.1f}%")
    else:
        print(f"✅ Inflation volatility is reasonable: {inflation_rates.std():.1f}%")
    
    if issues:
        print("\n⚠️  Issues detected:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ All checks passed! Model is generating realistic data.")
        return True

if __name__ == "__main__":
    success = test_mesa_model()
    sys.exit(0 if success else 1)
