#!/usr/bin/env python3
"""
Start the backend server with FRED integration.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(project_root)

import uvicorn
from src.api.main import app

if __name__ == "__main__":
    print("🏦 Starting EconAgent-Light Backend with FRED Integration...")
    print("📊 API Documentation: http://localhost:8000/api/docs")
    print("🔗 Health Check: http://localhost:8000/api/health")
    print("📈 FRED Data: http://localhost:8000/api/fred/current")
    print()
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )