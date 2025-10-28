@echo off
REM Start EconAgent-Light Web UI (Windows)
REM Launches Streamlit dashboard for running simulations and viewing results

echo 🚀 Starting EconAgent-Light Web UI...
echo ==================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not installed.
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo 📥 Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Create results directory
if not exist "web_results" mkdir web_results

REM Start Streamlit app
echo 🌐 Starting web interface...
echo.
echo 🎯 EconAgent-Light Dashboard will open in your browser
echo 📊 URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ==================================

streamlit run app.py --server.port 8501 --server.address localhost