"""
Comprehensive validation script for the Risk Dashboard.
Tests all components with real data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
os.environ['FRED_API_KEY'] = 'bcc1a43947af1745a35bfb3b7132b7c6'

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    try:
        import dash
        import plotly
        import pandas as pd
        import numpy as np
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        import mesa
        print("✓ All dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_fred_client():
    """Test FRED data fetching."""
    print("\nTesting FRED client...")
    try:
        from src.data.fred_client import FREDClient
        
        client = FREDClient()
        data = client.fetch_series('UNRATE', '2023-01-01', '2023-12-31')
        
        assert data is not None
        assert len(data) > 0
        print(f"✓ FRED client working - fetched {len(data)} data points")
        return True
    except Exception as e:
        print(f"✗ FRED client failed: {e}")
        return False

def test_data_pipeline():
    """Test data pipeline."""
    print("\nTesting data pipeline...")
    try:
        from src.data.fred_client import FREDClient
        from src.data.pipeline import DataPipeline, MissingValueHandler, FrequencyAligner
        from src.data.data_models import SeriesConfig
        
        fred_client = FREDClient()
        pipeline = DataPipeline(fred_client)
        pipeline.add_transformer(MissingValueHandler(method='ffill'))
        pipeline.add_transformer(FrequencyAligner(target_frequency='M'))
        
        series_config = {
            'unemployment': SeriesConfig(
                series_id='UNRATE',
                name='Unemployment Rate',
                start_date='2023-01-01',
                end_date='2023-12-31',
                frequency='monthly'
            )
        }
        
        data = pipeline.process(series_config)
        assert data is not None
        assert 'unemployment' in data.columns
        print(f"✓ Data pipeline working - processed {len(data)} rows")
        return True
    except Exception as e:
        print(f"✗ Data pipeline failed: {e}")
        return False

def test_forecasters():
    """Test forecasting models."""
    print("\nTesting forecasters...")
    try:
        import numpy as np
        import pandas as pd
        from src.models.arima_forecaster import ARIMAForecaster
        from src.models.ets_forecaster import ETSForecaster
        
        # Create sample data with DatetimeIndex
        dates = pd.date_range(start='2020-01-01', periods=50, freq='M')
        data = pd.Series(np.random.randn(50).cumsum() + 100, index=dates)
        
        # Test ARIMA
        arima = ARIMAForecaster(auto_order=False, order=(1,1,1))
        arima.fit(data)
        arima_forecast = arima.forecast(horizon=12)
        assert len(arima_forecast.point_forecast) == 12
        print(f"✓ ARIMA forecaster working - generated {len(arima_forecast.point_forecast)} forecasts")
        
        # Test ETS
        ets = ETSForecaster(trend='add', seasonal=None)
        ets.fit(data)
        ets_forecast = ets.forecast(horizon=12)
        assert len(ets_forecast.point_forecast) == 12
        print(f"✓ ETS forecaster working - generated {len(ets_forecast.point_forecast)} forecasts")
        
        return True
    except Exception as e:
        print(f"✗ Forecasters failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kri_calculator():
    """Test KRI calculator."""
    print("\nTesting KRI calculator...")
    try:
        import pandas as pd
        import numpy as np
        from src.kri.calculator import KRICalculator
        
        calc = KRICalculator()
        
        # Create sample forecast data
        forecasts = pd.DataFrame({
            'unemployment': np.random.uniform(3, 6, 12),
            'inflation': np.random.uniform(0.01, 0.03, 12),
            'interest_rate': np.random.uniform(2, 5, 12),
            'credit_spread': np.random.uniform(1, 3, 12)
        })
        
        kris = calc.compute_all_kris(forecasts=forecasts)
        assert len(kris) > 0
        print(f"✓ KRI calculator working - computed {len(kris)} KRIs")
        
        risk_levels = calc.evaluate_thresholds(kris)
        assert len(risk_levels) == len(kris)
        print(f"✓ Risk level evaluation working")
        
        return True
    except Exception as e:
        print(f"✗ KRI calculator failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simulation():
    """Test simulation model."""
    print("\nTesting simulation model...")
    try:
        from src.simulation.model import RiskSimulationModel
        from src.simulation.scenarios import BaselineScenario
        
        scenario = BaselineScenario()
        model = RiskSimulationModel(n_banks=3, n_firms=10, scenario=scenario)
        results = model.run_simulation(n_steps=10)
        
        assert results is not None
        assert len(results) == 10
        assert 'default_rate' in results.columns
        print(f"✓ Simulation working - ran {len(results)} steps")
        return True
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard_structure():
    """Test dashboard structure."""
    print("\nTesting dashboard structure...")
    try:
        from src.dashboard.app import app, data_cache
        
        assert app is not None
        assert app.layout is not None
        assert data_cache is not None
        print(f"✓ Dashboard structure valid")
        return True
    except Exception as e:
        print(f"✗ Dashboard structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("RISK DASHBOARD VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("FRED Client", test_fred_client),
        ("Data Pipeline", test_data_pipeline),
        ("Forecasters", test_forecasters),
        ("KRI Calculator", test_kri_calculator),
        ("Simulation", test_simulation),
        ("Dashboard Structure", test_dashboard_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Dashboard is ready to run!")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Please review errors above")
        return 1

if __name__ == '__main__':
    sys.exit(main())
