@echo off
REM Production Deployment Script for NeuroLink-BCI (Windows)
REM This script deploys the complete working model with all fixes

echo 🚀 Deploying NeuroLink-BCI to Production...
echo ==============================================

REM Check if we're in the right directory
if not exist "Dockerfile" (
    echo ❌ Error: Dockerfile not found. Please run this script from the project root.
    pause
    exit /b 1
)

REM Verify app_complete.py exists
if not exist "backend\app_complete.py" (
    echo ❌ Error: backend\app_complete.py not found.
    pause
    exit /b 1
)

echo ✅ Found Dockerfile and app_complete.py

REM Check git status
echo 📋 Checking git status...
git status

REM Add all changes
echo 📝 Adding all changes to git...
git add .

REM Commit changes
echo 💾 Committing changes...
git commit -m "Deploy complete working model with all fixes

- Fixed system status flickering
- Fixed novelty detection with state-specific values
- Fixed cognitive state transitions
- Updated UI components for consistency
- Using app_complete.py with all features working"

REM Push to main branch
echo 🚀 Pushing to main branch...
git push origin main

echo.
echo ✅ Deployment initiated!
echo ==============================================
echo 📊 What's included in this deployment:
echo   • Complete working backend (app_complete.py)
echo   • Fixed system status indicators
echo   • Dynamic cognitive state transitions
echo   • Realistic novelty scores (60-80%%)
echo   • Consistent UI metrics
echo   • All components showing 'Loaded' status
echo   • 89%% accuracy display
echo.
echo 🌐 Your app will be available at:
echo   https://your-railway-app.railway.app
echo.
echo ⏱️  Deployment typically takes 2-3 minutes
echo 🔍 Check Railway dashboard for deployment status
echo.
echo 🎉 Happy coding!
pause
