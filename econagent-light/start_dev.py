#!/usr/bin/env python3
"""
Development startup script for EconAgent-Light.
Starts both the FastAPI backend and React frontend in development mode.
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

def run_command(command, cwd=None, name="Process"):
    """Run a command and stream output."""
    print(f"Starting {name}...")
    print(f"Command: {' '.join(command)}")
    print(f"Working directory: {cwd or os.getcwd()}")
    print("-" * 50)
    
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            print(f"[{name}] {line.rstrip()}")
        
        process.wait()
        return process.returncode
        
    except KeyboardInterrupt:
        print(f"\nStopping {name}...")
        process.terminate()
        return 0
    except Exception as e:
        print(f"Error running {name}: {e}")
        return 1

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    # Check Python dependencies
    try:
        import fastapi
        import uvicorn
        import pandas
        import requests
        print("✓ Python dependencies found")
    except ImportError as e:
        print(f"✗ Missing Python dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Check Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Node.js found: {result.stdout.strip()}")
        else:
            print("✗ Node.js not found")
            return False
    except FileNotFoundError:
        print("✗ Node.js not found")
        print("Please install Node.js from https://nodejs.org/")
        return False
    
    # Check npm
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ npm found: {result.stdout.strip()}")
        else:
            print("✗ npm not found")
            return False
    except FileNotFoundError:
        print("✗ npm not found")
        return False
    
    return True

def setup_frontend():
    """Setup frontend dependencies if needed."""
    frontend_dir = Path(__file__).parent / "frontend"
    node_modules = frontend_dir / "node_modules"
    
    if not node_modules.exists():
        print("Installing frontend dependencies...")
        result = subprocess.run(
            ['npm', 'install'],
            cwd=frontend_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Failed to install frontend dependencies: {result.stderr}")
            return False
        
        print("✓ Frontend dependencies installed")
    else:
        print("✓ Frontend dependencies already installed")
    
    return True

def start_backend():
    """Start the FastAPI backend server."""
    backend_command = [
        sys.executable, '-m', 'uvicorn',
        'src.api.main:app',
        '--host', '0.0.0.0',
        '--port', '8000',
        '--reload',
        '--log-level', 'info'
    ]
    
    return run_command(backend_command, name="Backend")

def start_frontend():
    """Start the React frontend development server."""
    frontend_dir = Path(__file__).parent / "frontend"
    
    # Set environment variables for React
    env = os.environ.copy()
    env['REACT_APP_API_URL'] = 'http://localhost:8000/api'
    env['BROWSER'] = 'none'  # Don't auto-open browser
    
    frontend_command = ['npm', 'start']
    
    try:
        process = subprocess.Popen(
            frontend_command,
            cwd=frontend_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            print(f"[Frontend] {line.rstrip()}")
        
        process.wait()
        return process.returncode
        
    except KeyboardInterrupt:
        print("\nStopping Frontend...")
        process.terminate()
        return 0
    except Exception as e:
        print(f"Error running Frontend: {e}")
        return 1

def main():
    """Main function to start the development environment."""
    print("=" * 60)
    print("EconAgent-Light Development Server")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup frontend
    if not setup_frontend():
        sys.exit(1)
    
    print("\nStarting development servers...")
    print("Backend will be available at: http://localhost:8000")
    print("Frontend will be available at: http://localhost:3000")
    print("API documentation at: http://localhost:8000/api/docs")
    print("\nPress Ctrl+C to stop both servers")
    print("-" * 60)
    
    # Start both servers in separate threads
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    frontend_thread = threading.Thread(target=start_frontend, daemon=True)
    
    try:
        backend_thread.start()
        time.sleep(2)  # Give backend time to start
        frontend_thread.start()
        
        # Wait for threads
        backend_thread.join()
        frontend_thread.join()
        
    except KeyboardInterrupt:
        print("\nShutting down development servers...")
        sys.exit(0)

if __name__ == "__main__":
    main()