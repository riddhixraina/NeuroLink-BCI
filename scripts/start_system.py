#!/usr/bin/env python3
"""
NeuroLink-BCI System Startup Script
Automatically starts the backend and frontend components.
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

class SystemManager:
    def __init__(self):
        self.processes = []
        self.running = False
        
    def start_backend(self):
        """Start the Flask backend server."""
        print("🚀 Starting Flask backend server...")
        
        backend_dir = Path(__file__).parent.parent / "backend"
        os.chdir(backend_dir)
        
        # Start Flask app
        process = subprocess.Popen([
            sys.executable, "app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        self.processes.append(process)
        print("✅ Backend server started on http://localhost:5000")
        
        return process
    
    def start_frontend(self):
        """Start the React frontend development server."""
        print("🚀 Starting React frontend server...")
        
        frontend_dir = Path(__file__).parent.parent / "frontend"
        os.chdir(frontend_dir)
        
        # Start React app
        process = subprocess.Popen([
            "npm", "start"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        self.processes.append(process)
        print("✅ Frontend server starting on http://localhost:3000")
        
        return process
    
    def check_dependencies(self):
        """Check if all dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        # Check Python dependencies
        backend_dir = Path(__file__).parent.parent / "backend"
        requirements_file = backend_dir / "requirements.txt"
        
        if requirements_file.exists():
            print("📦 Python dependencies found")
        else:
            print("⚠️  Python requirements.txt not found")
        
        # Check Node.js dependencies
        frontend_dir = Path(__file__).parent.parent / "frontend"
        package_json = frontend_dir / "package.json"
        
        if package_json.exists():
            print("📦 Node.js dependencies found")
        else:
            print("⚠️  Node.js package.json not found")
    
    def install_dependencies(self):
        """Install missing dependencies."""
        print("📦 Installing dependencies...")
        
        # Install Python dependencies
        backend_dir = Path(__file__).parent.parent / "backend"
        if (backend_dir / "requirements.txt").exists():
            print("Installing Python dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         cwd=backend_dir, check=True)
        
        # Install Node.js dependencies
        frontend_dir = Path(__file__).parent.parent / "frontend"
        if (frontend_dir / "package.json").exists():
            print("Installing Node.js dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
    
    def start_system(self, install_deps=False):
        """Start the complete system."""
        print("🎯 Starting NeuroLink-BCI System")
        print("=" * 50)
        
        if install_deps:
            self.install_dependencies()
        
        self.check_dependencies()
        
        # Start backend
        backend_process = self.start_backend()
        
        # Wait for backend to start
        print("⏳ Waiting for backend to initialize...")
        time.sleep(3)
        
        # Start frontend
        frontend_process = self.start_frontend()
        
        self.running = True
        
        print("\n" + "=" * 50)
        print("🎉 NeuroLink-BCI System Started Successfully!")
        print("=" * 50)
        print("📊 Backend API: http://localhost:5000")
        print("🖥️  Frontend UI: http://localhost:3000")
        print("📚 API Docs: http://localhost:5000/api/status")
        print("\n💡 Press Ctrl+C to stop the system")
        print("=" * 50)
        
        return backend_process, frontend_process
    
    def stop_system(self):
        """Stop all running processes."""
        print("\n🛑 Stopping NeuroLink-BCI System...")
        
        self.running = False
        
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                print(f"Error stopping process: {e}")
        
        print("✅ System stopped successfully")
    
    def monitor_processes(self):
        """Monitor running processes."""
        while self.running:
            for i, process in enumerate(self.processes):
                if process.poll() is not None:
                    print(f"⚠️  Process {i} has stopped unexpectedly")
                    self.running = False
                    break
            time.sleep(1)

def signal_handler(signum, frame):
    """Handle Ctrl+C signal."""
    print("\n🛑 Received interrupt signal...")
    system_manager.stop_system()
    sys.exit(0)

def main():
    """Main function."""
    global system_manager
    
    # Parse command line arguments
    install_deps = "--install" in sys.argv or "-i" in sys.argv
    
    # Create system manager
    system_manager = SystemManager()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the system
        backend_process, frontend_process = system_manager.start_system(install_deps)
        
        # Monitor processes
        system_manager.monitor_processes()
        
    except KeyboardInterrupt:
        print("\n🛑 Received keyboard interrupt...")
    except Exception as e:
        print(f"❌ Error starting system: {e}")
    finally:
        system_manager.stop_system()

if __name__ == "__main__":
    main()
