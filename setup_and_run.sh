#!/bin/bash
# Setup and run the US Financial Risk Forecasting System

set -e

echo "=========================================="
echo "US Financial Risk Forecasting System"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Check if dependencies are installed
echo "Checking dependencies..."
if ! python -c "import dash, pandas, plotly, fredapi" 2>/dev/null; then
    echo "Installing dependencies (this may take a few minutes)..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

# Create necessary directories
echo "Setting up directories..."
mkdir -p data/cache data/processed logs
echo "✓ Directories ready"

# Check FRED API key
if grep -q "your_fred_api_key_here" .env 2>/dev/null; then
    echo ""
    echo "⚠️  WARNING: FRED API key not configured!"
    echo "Please edit .env file and add your FRED API key"
    echo "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
    echo ""
fi

echo ""
echo "=========================================="
echo "Starting Dashboard..."
echo "=========================================="
echo ""
echo "Dashboard will be available at: http://localhost:8050"
echo "Press Ctrl+C to stop"
echo ""

# Run the dashboard
python src/dashboard/app.py
