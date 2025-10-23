#!/bin/bash
# Multi-Platform Deployment Script for NeuroLink-BCI

set -e

echo "🚀 NeuroLink-BCI Multi-Platform Deployment"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[DEPLOY]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "backend/app.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Function to deploy to Vercel
deploy_vercel() {
    print_header "Deploying Frontend to Vercel..."
    
    if ! command -v vercel &> /dev/null; then
        print_warning "Vercel CLI not found. Installing..."
        npm install -g vercel
    fi
    
    cd frontend
    print_status "Building frontend..."
    npm run build
    
    print_status "Deploying to Vercel..."
    vercel --prod
    
    print_status "Frontend deployed to Vercel!"
    cd ..
}

# Function to deploy to Railway
deploy_railway() {
    print_header "Deploying Backend to Railway..."
    
    if ! command -v railway &> /dev/null; then
        print_warning "Railway CLI not found. Installing..."
        npm install -g @railway/cli
    fi
    
    print_status "Logging into Railway..."
    railway login
    
    print_status "Creating Railway project..."
    railway init
    
    print_status "Setting environment variables..."
    railway variables set FLASK_ENV=production
    railway variables set SECRET_KEY=$(openssl rand -hex 32)
    
    print_status "Deploying to Railway..."
    railway up
    
    print_status "Backend deployed to Railway!"
}

# Function to deploy to Netlify
deploy_netlify() {
    print_header "Deploying Frontend to Netlify..."
    
    if ! command -v netlify &> /dev/null; then
        print_warning "Netlify CLI not found. Installing..."
        npm install -g netlify-cli
    fi
    
    cd frontend
    print_status "Building frontend..."
    npm run build
    
    print_status "Deploying to Netlify..."
    netlify deploy --prod --dir=build
    
    print_status "Frontend deployed to Netlify!"
    cd ..
}

# Function to deploy to Heroku
deploy_heroku() {
    print_header "Deploying Backend to Heroku..."
    
    if ! command -v heroku &> /dev/null; then
        print_error "Heroku CLI not found. Please install from https://devcenter.heroku.com/articles/heroku-cli"
        exit 1
    fi
    
    print_status "Logging into Heroku..."
    heroku login
    
    print_status "Creating Heroku app..."
    heroku create neuralink-bci-$(date +%s)
    
    print_status "Setting environment variables..."
    heroku config:set FLASK_ENV=production
    heroku config:set SECRET_KEY=$(openssl rand -hex 32)
    
    print_status "Deploying to Heroku..."
    git push heroku main
    
    print_status "Backend deployed to Heroku!"
}

# Function to deploy to Render
deploy_render() {
    print_header "Deploying to Render..."
    
    print_status "Creating render.yaml configuration..."
    cat > render.yaml << 'EOF'
services:
  - type: web
    name: neuralink-bci-backend
    env: python
    buildCommand: pip install -r backend/requirements-prod.txt
    startCommand: gunicorn --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker --bind 0.0.0.0:$PORT backend.app:app
    envVars:
      - key: FLASK_ENV
        value: production
      - key: SECRET_KEY
        generateValue: true
    healthCheckPath: /api/health

  - type: web
    name: neuralink-bci-frontend
    env: static
    buildCommand: cd frontend && npm install && npm run build
    staticPublishPath: frontend/build
EOF
    
    print_status "Please manually deploy to Render using the render.yaml file"
    print_status "Go to https://render.com and create a new service"
}

# Function to deploy with Docker
deploy_docker() {
    print_header "Deploying with Docker..."
    
    print_status "Building Docker images..."
    docker build -t neuralink-bci-backend .
    docker build -t neuralink-bci-frontend ./frontend
    
    print_status "Running with Docker Compose..."
    docker-compose up -d
    
    print_status "Docker deployment complete!"
    print_status "Access at: http://localhost:80"
}

# Main deployment menu
echo "Select deployment platform:"
echo "1) Vercel (Frontend) + Railway (Backend)"
echo "2) Netlify (Frontend) + Heroku (Backend)"
echo "3) Render (Full Stack)"
echo "4) Docker (Local)"
echo "5) All platforms (for testing)"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        deploy_vercel
        deploy_railway
        ;;
    2)
        deploy_netlify
        deploy_heroku
        ;;
    3)
        deploy_render
        ;;
    4)
        deploy_docker
        ;;
    5)
        deploy_vercel
        deploy_railway
        deploy_netlify
        deploy_heroku
        deploy_render
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

print_header "Deployment completed!"
echo ""
echo "📋 Next steps:"
echo "1. Update frontend environment variables with backend URLs"
echo "2. Test all endpoints"
echo "3. Configure custom domains (optional)"
echo "4. Set up monitoring and alerts"
echo ""
echo "🎉 Your NeuroLink-BCI is now live!"
