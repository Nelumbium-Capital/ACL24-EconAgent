#!/usr/bin/env python3
"""
Test script to verify EconAgent-Light system functionality.
Tests FRED integration, API endpoints, and configuration.
"""

import os
import sys
import asyncio
import requests
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_environment():
    """Test environment variables and configuration."""
    print("🔧 Testing Environment Configuration...")
    
    # Check .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    print("✅ .env file found")
    
    # Load and check configuration
    try:
        from config import DEFAULT_CONFIG
        print(f"✅ Configuration loaded successfully")
        print(f"   FRED API Key: {'***' + DEFAULT_CONFIG.fred.api_key[-4:] if DEFAULT_CONFIG.fred.api_key else 'Not set'}")
        print(f"   Cache Directory: {DEFAULT_CONFIG.fred.cache_dir}")
        print(f"   Log Level: {DEFAULT_CONFIG.log_level}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_fred_client():
    """Test FRED client functionality."""
    print("\n📊 Testing FRED Client...")
    
    try:
        from src.data_integration.fred_client import FREDClient
        from config import DEFAULT_CONFIG
        
        # Initialize client
        fred_client = FREDClient(
            api_key=DEFAULT_CONFIG.fred.api_key,
            cache_dir=DEFAULT_CONFIG.fred.cache_dir,
            cache_hours=DEFAULT_CONFIG.fred.cache_hours
        )
        
        print("✅ FRED client initialized")
        
        # Test connection
        if fred_client._validate_connection():
            print("✅ FRED API connection successful")
        else:
            print("⚠️  FRED API connection failed (may work with limited functionality)")
        
        # Test getting current economic snapshot
        snapshot = fred_client.get_current_economic_snapshot()
        print("✅ Economic snapshot retrieved")
        print(f"   Unemployment Rate: {snapshot.unemployment_rate:.1f}%")
        print(f"   Inflation Rate: {snapshot.inflation_rate:.1f}%")
        print(f"   Fed Funds Rate: {snapshot.fed_funds_rate:.1f}%")
        
        # Test statistics
        stats = fred_client.get_statistics()
        print(f"✅ Client statistics: {stats['total_requests']} requests, {stats['cache_hit_rate']} cache hit rate")
        
        return True
        
    except Exception as e:
        print(f"❌ FRED client error: {e}")
        return False

def test_calibration_engine():
    """Test calibration engine functionality."""
    print("\n⚙️ Testing Calibration Engine...")
    
    try:
        from src.data_integration.fred_client import FREDClient
        from src.data_integration.calibration_engine import CalibrationEngine
        from config import DEFAULT_CONFIG
        
        # Initialize components
        fred_client = FREDClient(
            api_key=DEFAULT_CONFIG.fred.api_key,
            cache_dir=DEFAULT_CONFIG.fred.cache_dir
        )
        calibration_engine = CalibrationEngine(fred_client)
        
        print("✅ Calibration engine initialized")
        
        # Test calibration
        result = calibration_engine.calibrate_simulation_parameters()
        print("✅ Calibration completed")
        print(f"   Unemployment Target: {result.unemployment_target:.1f}%")
        print(f"   Inflation Target: {result.inflation_target:.1f}%")
        print(f"   Confidence Score: {result.confidence_score:.2f}")
        
        # Test summary
        summary = calibration_engine.get_calibration_summary()
        print(f"✅ Calibration summary: {summary['status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Calibration engine error: {e}")
        return False

def test_api_server():
    """Test if API server can start and respond."""
    print("\n🌐 Testing API Server...")
    
    try:
        # Import FastAPI app
        from src.api.main import app
        print("✅ FastAPI app imported successfully")
        
        # Test that endpoints are registered
        routes = [route.path for route in app.routes]
        expected_routes = ["/api/health", "/api/fred/current", "/api/simulations/"]
        
        for route in expected_routes:
            if any(route in r for r in routes):
                print(f"✅ Route {route} registered")
            else:
                print(f"⚠️  Route {route} not found")
        
        return True
        
    except Exception as e:
        print(f"❌ API server error: {e}")
        return False

def test_frontend_setup():
    """Test frontend setup and dependencies."""
    print("\n⚛️ Testing Frontend Setup...")
    
    frontend_dir = project_root / "frontend"
    
    # Check if frontend directory exists
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    print("✅ Frontend directory found")
    
    # Check package.json
    package_json = frontend_dir / "package.json"
    if package_json.exists():
        print("✅ package.json found")
    else:
        print("❌ package.json not found")
        return False
    
    # Check node_modules
    node_modules = frontend_dir / "node_modules"
    if node_modules.exists():
        print("✅ node_modules found (dependencies installed)")
    else:
        print("⚠️  node_modules not found (run 'npm install' in frontend directory)")
    
    # Check key files
    key_files = ["src/App.tsx", "src/index.tsx", "tailwind.config.js"]
    for file in key_files:
        if (frontend_dir / file).exists():
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} not found")
    
    return True

def test_dependencies():
    """Test Python dependencies."""
    print("\n📦 Testing Python Dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "requests",
        "pydantic",
        "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("EconAgent-Light System Test")
    print("=" * 60)
    
    tests = [
        ("Environment", test_environment),
        ("Dependencies", test_dependencies),
        ("FRED Client", test_fred_client),
        ("Calibration Engine", test_calibration_engine),
        ("API Server", test_api_server),
        ("Frontend Setup", test_frontend_setup),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run: python start_dev.py")
        print("2. Open: http://localhost:3000")
        print("3. Create your first simulation!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)