#!/usr/bin/env python3
"""
Simple FRED API test without numpy dependencies.
Tests basic FRED data fetching functionality.
"""

import sys
import logging
import requests
from datetime import datetime

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_fred_api_direct():
    """Test FRED API directly without dependencies."""
    logger = logging.getLogger(__name__)
    
    print("=" * 70)
    print("TESTING FRED API INTEGRATION (Direct)")
    print("=" * 70)
    
    # Your API key
    api_key = "bcc1a43947af1745a35bfb3b7132b7c6"
    base_url = "https://api.stlouisfed.org/fred"
    
    print(f"Using FRED API Key: {api_key[:8]}...{api_key[-4:]}")
    
    # Test series to fetch
    test_series = {
        'UNRATE': 'Unemployment Rate',
        'FEDFUNDS': 'Federal Funds Rate',
        'CPIAUCSL': 'Consumer Price Index',
        'GDP': 'Gross Domestic Product'
    }
    
    success_count = 0
    
    for series_id, description in test_series.items():
        try:
            logger.info(f"Testing {series_id} ({description})...")
            
            # Build request URL
            url = f"{base_url}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': api_key,
                'file_type': 'json',
                'limit': 5,  # Just get last 5 observations
                'sort_order': 'desc'  # Most recent first
            }
            
            # Make request
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'observations' in data and len(data['observations']) > 0:
                latest_obs = data['observations'][0]
                value = latest_obs['value']
                date = latest_obs['date']
                
                if value != '.':  # FRED uses '.' for missing values
                    print(f"✅ {series_id}: {value} (as of {date})")
                    success_count += 1
                else:
                    print(f"⚠️  {series_id}: No recent data available")
            else:
                print(f"❌ {series_id}: No data returned")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {series_id}: Network error - {e}")
        except Exception as e:
            print(f"❌ {series_id}: Error - {e}")
    
    print(f"\nResults: {success_count}/{len(test_series)} series successfully fetched")
    
    if success_count >= 3:
        print("✅ FRED API integration is working!")
        return True
    else:
        print("❌ FRED API integration has issues")
        return False

def test_fred_client_basic():
    """Test basic FRED client functionality without numpy."""
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 70)
    print("TESTING FRED CLIENT CLASS (Basic)")
    print("=" * 70)
    
    try:
        # Import just the FRED client
        sys.path.insert(0, './src')
        
        # Test basic imports
        try:
            import pandas as pd
            print("✅ Pandas available")
        except ImportError:
            print("❌ Pandas not available")
            return False
        
        try:
            from data_integration.fred_client import FREDClient
            print("✅ FREDClient import successful")
        except ImportError as e:
            print(f"❌ FREDClient import failed: {e}")
            return False
        
        # Initialize client
        api_key = "bcc1a43947af1745a35bfb3b7132b7c6"
        
        fred = FREDClient(api_key=api_key, cache_dir="./test_cache")
        print("✅ FREDClient initialized")
        
        # Test fetching a simple series
        logger.info("Testing unemployment rate fetch...")
        
        unemployment_data = fred.get_series('UNRATE', start_date='2023-01-01')
        
        if not unemployment_data.empty:
            latest_value = unemployment_data.iloc[-1, 0]
            latest_date = unemployment_data.index[-1].strftime("%Y-%m-%d")
            print(f"✅ Unemployment data: {latest_value:.1f}% (as of {latest_date})")
            print(f"   Total observations: {len(unemployment_data)}")
        else:
            print("❌ No unemployment data retrieved")
            return False
        
        # Test series info
        logger.info("Testing series info...")
        
        series_info = fred.get_series_info('UNRATE')
        if series_info:
            title = series_info.get('title', 'Unknown')
            units = series_info.get('units', 'Unknown')
            print(f"✅ Series info: {title} ({units})")
        else:
            print("⚠️  Series info not available")
        
        print("✅ FRED CLIENT TEST SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ FRED client test failed: {e}")
        logger.exception("Full error details:")
        return False

def main():
    """Main test function."""
    setup_logging()
    
    success = True
    
    try:
        # Test direct API access
        if not test_fred_api_direct():
            success = False
        
        # Test FRED client class
        if not test_fred_client_basic():
            success = False
        
        if success:
            print("\n🎉 ALL BASIC TESTS PASSED!")
            print("🎉 FRED API integration is working")
            print("🎉 Your API key is valid and functional")
            print("\nNext steps:")
            print("- Install compatible numpy/pandas for full functionality")
            print("- Run the full simulation with: python3 demo.py")
        else:
            print("\n❌ SOME TESTS FAILED")
            print("❌ Check your internet connection and API key")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()