"""
Quick test to validate dashboard imports and basic functionality.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_dashboard_imports():
    """Test that all dashboard imports work."""
    try:
        from src.dashboard.app import app, fetch_and_process_data
        print("✓ Dashboard imports successful")
        return True
    except Exception as e:
        print(f"✗ Dashboard import failed: {e}")
        return False

def test_dashboard_layout():
    """Test that dashboard layout is defined."""
    try:
        from src.dashboard.app import app
        assert app.layout is not None
        print("✓ Dashboard layout defined")
        return True
    except Exception as e:
        print(f"✗ Dashboard layout test failed: {e}")
        return False

def test_callbacks_registered():
    """Test that callbacks are registered."""
    try:
        from src.dashboard.app import app
        # Check that callbacks exist
        assert len(app.callback_map) > 0
        print(f"✓ Dashboard has {len(app.callback_map)} callbacks registered")
        return True
    except Exception as e:
        print(f"✗ Callback registration test failed: {e}")
        return False

if __name__ == '__main__':
    print("Testing Risk Dashboard Implementation...")
    print("=" * 50)
    
    tests = [
        test_dashboard_imports,
        test_dashboard_layout,
        test_callbacks_registered
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("=" * 50)
    if all(results):
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
