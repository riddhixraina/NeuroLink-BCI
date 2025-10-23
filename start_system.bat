@echo off
echo Starting NeuroLink-BCI System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed or not in PATH
    echo Please install Node.js 16+ from https://nodejs.org
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist "backend\venv" (
    echo Creating Python virtual environment...
    python -m venv backend\venv
)

REM Activate virtual environment
call backend\venv\Scripts\activate.bat

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r backend\requirements.txt

REM Install Node.js dependencies
echo Installing Node.js dependencies...
cd frontend
npm install
cd ..

REM Start the system
echo Starting NeuroLink-BCI System...
python scripts\start_system.py

pause
