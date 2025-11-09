"""
Validation script for classical time-series forecasting models.
Checks syntax and basic structure without requiring dependencies.
"""
import sys
from pathlib import Path
import ast

def validate_python_file(filepath: Path) -> tuple[bool, str]:
    """
    Validate a Python file for syntax errors.
    
    Args:
        filepath: Path to Python file
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        
        # Try to parse the file
        ast.parse(code)
        return True, "Valid Python syntax"
    
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def check_class_definitions(filepath: Path, expected_classes: list) -> tuple[bool, str]:
    """
    Check if expected classes are defined in the file.
    
    Args:
        filepath: Path to Python file
        expected_classes: List of expected class names
        
    Returns:
        Tuple of (all_found, message)
    """
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        # Find all class definitions
        defined_classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        # Check if all expected classes are present
        missing = [cls for cls in expected_classes if cls not in defined_classes]
        
        if missing:
            return False, f"Missing classes: {', '.join(missing)}"
        
        return True, f"All expected classes found: {', '.join(expected_classes)}"
    
    except Exception as e:
        return False, f"Error checking classes: {str(e)}"


def check_method_definitions(filepath: Path, class_name: str, expected_methods: list) -> tuple[bool, str]:
    """
    Check if expected methods are defined in a class.
    
    Args:
        filepath: Path to Python file
        class_name: Name of the class to check
        expected_methods: List of expected method names
        
    Returns:
        Tuple of (all_found, message)
    """
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        
        tree = ast.parse(code)
        
        # Find the class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Get all method names
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                
                # Check if all expected methods are present
                missing = [m for m in expected_methods if m not in methods]
                
                if missing:
                    return False, f"Missing methods in {class_name}: {', '.join(missing)}"
                
                return True, f"All expected methods found in {class_name}"
        
        return False, f"Class {class_name} not found"
    
    except Exception as e:
        return False, f"Error checking methods: {str(e)}"


def main():
    """Run validation checks."""
    print("\n" + "="*70)
    print("Classical Time-Series Forecasting Models Validation")
    print("="*70)
    
    base_path = Path(__file__).parent.parent / "src" / "models"
    
    # Define validation checks
    checks = [
        {
            "name": "Base Forecaster Syntax",
            "file": base_path / "base_forecaster.py",
            "validator": validate_python_file
        },
        {
            "name": "Base Forecaster Classes",
            "file": base_path / "base_forecaster.py",
            "validator": lambda f: check_class_definitions(f, ["BaseForecaster", "ForecastResult"])
        },
        {
            "name": "Base Forecaster Methods",
            "file": base_path / "base_forecaster.py",
            "validator": lambda f: check_method_definitions(f, "BaseForecaster", ["fit", "forecast", "validate_data"])
        },
        {
            "name": "ARIMA Forecaster Syntax",
            "file": base_path / "arima_forecaster.py",
            "validator": validate_python_file
        },
        {
            "name": "ARIMA Forecaster Classes",
            "file": base_path / "arima_forecaster.py",
            "validator": lambda f: check_class_definitions(f, ["ARIMAForecaster", "SARIMAForecaster"])
        },
        {
            "name": "ARIMA Forecaster Methods",
            "file": base_path / "arima_forecaster.py",
            "validator": lambda f: check_method_definitions(f, "ARIMAForecaster", ["fit", "forecast", "_auto_select_order"])
        },
        {
            "name": "ETS Forecaster Syntax",
            "file": base_path / "ets_forecaster.py",
            "validator": validate_python_file
        },
        {
            "name": "ETS Forecaster Classes",
            "file": base_path / "ets_forecaster.py",
            "validator": lambda f: check_class_definitions(f, ["ETSForecaster", "SimpleExponentialSmoothing", "HoltLinearTrend", "HoltWinters", "AutoETS"])
        },
        {
            "name": "ETS Forecaster Methods",
            "file": base_path / "ets_forecaster.py",
            "validator": lambda f: check_method_definitions(f, "ETSForecaster", ["fit", "forecast", "get_components"])
        },
        {
            "name": "Models __init__ Syntax",
            "file": base_path / "__init__.py",
            "validator": validate_python_file
        }
    ]
    
    results = {}
    
    for check in checks:
        print(f"\n{check['name']}...")
        
        if not check['file'].exists():
            results[check['name']] = ("FAILED", f"File not found: {check['file']}")
            print(f"  ✗ FAILED: File not found")
            continue
        
        is_valid, message = check['validator'](check['file'])
        
        if is_valid:
            results[check['name']] = ("PASSED", message)
            print(f"  ✓ PASSED: {message}")
        else:
            results[check['name']] = ("FAILED", message)
            print(f"  ✗ FAILED: {message}")
    
    # Print summary
    print("\n" + "="*70)
    print("Validation Summary")
    print("="*70)
    
    passed = sum(1 for status, _ in results.values() if status == "PASSED")
    total = len(results)
    
    for check_name, (status, message) in results.items():
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {check_name}")
    
    print("\n" + "="*70)
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("ALL VALIDATIONS PASSED ✓")
        print("\nImplementation Summary:")
        print("  • BaseForecaster: Abstract base class with common utilities")
        print("  • ARIMAForecaster: ARIMA model with automatic order selection")
        print("  • SARIMAForecaster: Seasonal ARIMA with seasonal decomposition")
        print("  • ETSForecaster: Exponential smoothing with trend/seasonal support")
        print("  • SimpleExponentialSmoothing: SES for stationary data")
        print("  • HoltLinearTrend: Holt's method for trended data")
        print("  • HoltWinters: Full seasonal exponential smoothing")
        print("  • AutoETS: Automatic model selection")
    else:
        print("SOME VALIDATIONS FAILED ✗")
    
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
