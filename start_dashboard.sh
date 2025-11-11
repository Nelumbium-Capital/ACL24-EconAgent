#!/bin/bash
# Start the Risk Forecasting Dashboard

cd "$(dirname "$0")"

echo "Starting Risk Forecasting Dashboard..."
echo "Dashboard will be available at: http://localhost:8050"
echo ""

# Run with the current Python environment
python3 src/dashboard/app.py
