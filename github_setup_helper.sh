#!/bin/bash
# Quick GitHub Setup Script for NeuroLink-BCI

echo " NeuroLink-BCI GitHub Setup Helper"
echo "====================================="
echo ""

# Check if git is configured
if ! git config user.name > /dev/null 2>&1; then
    echo "  Git user not configured. Please run:"
    echo "   git config --global user.name 'Your Name'"
    echo "   git config --global user.email 'your.email@example.com'"
    echo ""
fi

echo " Next Steps:"
echo "1. Go to https://github.com/new"
echo "2. Repository name: NeuroLink-BCI"
echo "3. Description: Real-Time Neural Decoding System with CNN-LSTM Architecture"
echo "4. Make it Public"
echo "5. DO NOT initialize with README (we have one)"
echo "6. Click 'Create repository'"
echo ""
echo "7. After creating, run these commands:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/NeuroLink-BCI.git"
echo "   git push -u origin main"
echo ""
echo "8. For live deployment, follow DEPLOYMENT_INSTRUCTIONS.md"
echo ""
echo " Your project is ready for GitHub!"
