#!/usr/bin/env bash

# Start EconAgent-Light Web UI
# Launches Streamlit dashboard for running simulations and viewing results

set -e

echo "🚀 Starting EconAgent-Light Web UI..."
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create results directory
mkdir -p web_results

# Start Streamlit app
echo "🌐 Starting web interface..."
echo ""
echo "🎯 EconAgent-Light Dashboard will open in your browser"
echo "📊 URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="

streamlit run app.py --server.port 8501 --server.address localhost