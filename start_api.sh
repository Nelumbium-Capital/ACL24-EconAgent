#!/bin/bash
# Start the Risk Forecasting API Server

cd "$(dirname "$0")"

# Use the correct python from venv
PYTHON="/Users/cameronmalloy/ACL24-EconAgent/econagent-light/venv_arm64/bin/python3"

echo "Starting Risk Forecasting API Server..."
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""

$PYTHON src/api/server.py
