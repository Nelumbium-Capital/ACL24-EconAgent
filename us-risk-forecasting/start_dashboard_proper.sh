#!/bin/bash
# Start the Risk Forecasting Dashboard

cd "$(dirname "$0")"

# Use the correct python from venv
PYTHON="/Users/cameronmalloy/ACL24-EconAgent/econagent-light/venv_arm64/bin/python3"

echo "Starting Risk Forecasting Dashboard..."
echo "Dashboard will be available at: http://localhost:8050"
echo ""

$PYTHON src/dashboard/app.py
