#!/bin/bash
# Quick Deployment Script for NeuroLink-BCI

echo "Starting quick deployment..."

# Build frontend
echo "Building frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Frontend build completed!"
echo "Ready for deployment to Vercel and Railway"

# Instructions
echo ""
echo "DEPLOYMENT INSTRUCTIONS:"
echo "1. Push to GitHub: git push origin main"
echo "2. Deploy frontend to Vercel:"
echo "   - Go to vercel.com"
echo "   - Import NeuroLink-BCI repository"
echo "   - Root Directory: frontend"
echo "   - Build Command: npm run build"
echo "   - Output Directory: build"
echo "3. Deploy backend to Railway:"
echo "   - Go to railway.app"
echo "   - New project from GitHub"
echo "   - Select NeuroLink-BCI"
echo "   - Add environment variables:"
echo "     FLASK_ENV=production"
echo "     SECRET_KEY=your-secret-key"
echo "4. Update frontend API URL in Vercel environment variables"
echo ""
echo "Your prototype will be live!"
