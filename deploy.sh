#!/bin/bash
# Production Deployment Script for NeuroLink-BCI

set -e  # Exit on any error

echo "🚀 Starting NeuroLink-BCI Production Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "backend/app.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs
mkdir -p models
mkdir -p uploads
mkdir -p static

# Set up Python virtual environment
print_status "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r backend/requirements-prod.txt

# Install development dependencies if in dev mode
if [ "$1" = "dev" ]; then
    print_status "Installing development dependencies..."
    pip install pytest pytest-flask black flake8
fi

# Set up environment variables
print_status "Setting up environment variables..."
if [ ! -f ".env" ]; then
    if [ -f "env.example" ]; then
        cp env.example .env
        print_warning "Created .env file from template. Please update with your production values."
    else
        print_warning "No .env file found. Please create one with your configuration."
    fi
fi

# Train the model if it doesn't exist
print_status "Checking for trained model..."
if [ ! -f "models/trained_model.pth" ]; then
    print_status "Training model..."
    python scripts/quick_train_model.py
else
    print_status "Trained model found."
fi

# Build frontend
print_status "Building frontend..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi

# Create production build
npm run build
cd ..

# Set up logging
print_status "Setting up logging..."
mkdir -p logs
touch logs/app.log

# Create systemd service file (for Linux)
if command -v systemctl &> /dev/null; then
    print_status "Creating systemd service file..."
    cat > neuralink-bci.service << EOF
[Unit]
Description=NeuroLink-BCI Backend Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
Environment=FLASK_ENV=production
ExecStart=$(pwd)/venv/bin/gunicorn --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker --bind 0.0.0.0:5000 backend.app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    print_status "Systemd service file created. To install:"
    echo "sudo cp neuralink-bci.service /etc/systemd/system/"
    echo "sudo systemctl daemon-reload"
    echo "sudo systemctl enable neuralink-bci"
    echo "sudo systemctl start neuralink-bci"
fi

# Create nginx configuration
print_status "Creating nginx configuration..."
cat > nginx.conf << EOF
server {
    listen 80;
    server_name your-domain.com;  # Change this to your domain

    # Frontend static files
    location / {
        root $(pwd)/frontend/build;
        try_files \$uri \$uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # WebSocket support
    location /socket.io/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
}
EOF

print_status "Nginx configuration created. To install:"
echo "sudo cp nginx.conf /etc/nginx/sites-available/neuralink-bci"
echo "sudo ln -s /etc/nginx/sites-available/neuralink-bci /etc/nginx/sites-enabled/"
echo "sudo nginx -t"
echo "sudo systemctl reload nginx"

# Create startup script
print_status "Creating startup script..."
cat > start_production.sh << 'EOF'
#!/bin/bash
# Production startup script for NeuroLink-BCI

# Activate virtual environment
source venv/bin/activate

# Set production environment
export FLASK_ENV=production

# Start the application
exec gunicorn --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
              --bind 0.0.0.0:5000 \
              --workers 4 \
              --timeout 120 \
              --keep-alive 2 \
              --max-requests 1000 \
              --max-requests-jitter 100 \
              --access-logfile logs/access.log \
              --error-logfile logs/error.log \
              backend.app:app
EOF

chmod +x start_production.sh

# Create health check script
print_status "Creating health check script..."
cat > health_check.sh << 'EOF'
#!/bin/bash
# Health check script for NeuroLink-BCI

# Check if the service is running
if curl -f http://localhost:5000/api/health > /dev/null 2>&1; then
    echo "✅ Service is healthy"
    exit 0
else
    echo "❌ Service is not responding"
    exit 1
fi
EOF

chmod +x health_check.sh

# Final status
print_status "Deployment completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Update .env file with your production values"
echo "2. Configure your domain in nginx.conf"
echo "3. Install and start the systemd service"
echo "4. Configure nginx and reload"
echo "5. Test the deployment with: ./health_check.sh"
echo ""
echo "🌐 Your application will be available at:"
echo "   Frontend: http://your-domain.com"
echo "   Backend API: http://your-domain.com/api/"
echo "   WebSocket: ws://your-domain.com/socket.io/"
echo ""
echo "📊 Monitor logs with:"
echo "   tail -f logs/app.log"
echo "   tail -f logs/access.log"
echo "   tail -f logs/error.log"
