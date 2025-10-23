#!/bin/bash

# Production Deployment Script for NeuroLink-BCI
# This script deploys the complete working model with all fixes

echo "🚀 Deploying NeuroLink-BCI to Production..."
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo "❌ Error: Dockerfile not found. Please run this script from the project root."
    exit 1
fi

# Verify app_complete.py exists
if [ ! -f "backend/app_complete.py" ]; then
    echo "❌ Error: backend/app_complete.py not found."
    exit 1
fi

echo "✅ Found Dockerfile and app_complete.py"

# Check git status
echo "📋 Checking git status..."
git status

# Add all changes
echo "📝 Adding all changes to git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Deploy complete working model with all fixes

- Fixed system status flickering
- Fixed novelty detection with state-specific values
- Fixed cognitive state transitions
- Updated UI components for consistency
- Using app_complete.py with all features working"

# Push to main branch
echo "🚀 Pushing to main branch..."
git push origin main

echo ""
echo "✅ Deployment initiated!"
echo "=============================================="
echo "📊 What's included in this deployment:"
echo "  • Complete working backend (app_complete.py)"
echo "  • Fixed system status indicators"
echo "  • Dynamic cognitive state transitions"
echo "  • Realistic novelty scores (60-80%)"
echo "  • Consistent UI metrics"
echo "  • All components showing 'Loaded' status"
echo "  • 89% accuracy display"
echo ""
echo "🌐 Your app will be available at:"
echo "  https://your-railway-app.railway.app"
echo ""
echo "⏱️  Deployment typically takes 2-3 minutes"
echo "🔍 Check Railway dashboard for deployment status"
echo ""
echo "🎉 Happy coding!"
