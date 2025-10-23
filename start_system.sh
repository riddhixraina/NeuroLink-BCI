#!/bin/bash

echo "Starting NeuroLink-BCI System..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    echo "Please install Node.js 16+ from https://nodejs.org"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed"
    echo "Please install npm (usually comes with Node.js)"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "backend/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv backend/venv
fi

# Activate virtual environment
source backend/venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r backend/requirements.txt

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
cd frontend
npm install
cd ..

# Make the start script executable
chmod +x scripts/start_system.py

# Start the system
echo "Starting NeuroLink-BCI System..."
python3 scripts/start_system.py
