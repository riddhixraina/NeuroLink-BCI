# 🚀 Deployment Guide for NeuroLink-BCI

This guide covers various deployment options for the NeuroLink-BCI system, from local development to production cloud deployment.

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- Git
- Docker (optional)
- Server with Ubuntu 18.04+ (for production)

## 🏠 Local Development

### Quick Start
```bash
# Clone and setup
git clone https://github.com/yourusername/NeuroLink-BCI.git
cd NeuroLink-BCI

# Backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r backend/requirements-prod.txt
python scripts/quick_train_model.py

# Frontend
cd frontend
npm install
npm start

# Backend (in another terminal)
cd ..
python backend/app.py
```

### Access Points
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **WebSocket**: ws://localhost:5000/socket.io/

## 🐳 Docker Deployment

### Single Container
```bash
# Build and run
docker build -t neuralink-bci .
docker run -p 5000:5000 neuralink-bci
```

### Multi-Container (Recommended)
```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Access Points
- **Application**: http://localhost:80
- **Backend API**: http://localhost:5000

## ☁️ Cloud Deployment

### AWS Deployment

#### EC2 Instance
```bash
# Launch Ubuntu 20.04 LTS instance
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Clone repository
git clone https://github.com/yourusername/NeuroLink-BCI.git
cd NeuroLink-BCI

# Run deployment script
chmod +x deploy.sh
./deploy.sh

# Start production server
./start_production.sh
```

#### ECS (Elastic Container Service)
```bash
# Build and push Docker image
docker build -t neuralink-bci .
docker tag neuralink-bci:latest your-account.dkr.ecr.region.amazonaws.com/neuralink-bci:latest
docker push your-account.dkr.ecr.region.amazonaws.com/neuralink-bci:latest

# Deploy using ECS CLI or AWS Console
```

### Google Cloud Platform

#### Compute Engine
```bash
# Create VM instance
gcloud compute instances create neuralink-bci \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --machine-type=e2-medium \
    --zone=us-central1-a

# SSH and deploy
gcloud compute ssh neuralink-bci
# Follow EC2 deployment steps
```

#### Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/neuralink-bci
gcloud run deploy --image gcr.io/PROJECT-ID/neuralink-bci --platform managed
```

### Azure

#### Virtual Machine
```bash
# Create VM using Azure CLI
az vm create \
    --resource-group myResourceGroup \
    --name neuralink-bci \
    --image UbuntuLTS \
    --admin-username azureuser \
    --generate-ssh-keys

# SSH and deploy
ssh azureuser@your-vm-ip
# Follow EC2 deployment steps
```

#### Container Instances
```bash
# Deploy container
az container create \
    --resource-group myResourceGroup \
    --name neuralink-bci \
    --image your-registry/neuralink-bci:latest \
    --ports 5000
```

## 🌐 Production Deployment

### Server Setup (Ubuntu)

#### 1. System Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3 python3-pip python3-venv nodejs npm nginx git curl

# Install Python 3.9+ if needed
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.9 python3.9-venv python3.9-dev
```

#### 2. Application Deployment
```bash
# Clone repository
git clone https://github.com/yourusername/NeuroLink-BCI.git
cd NeuroLink-BCI

# Run deployment script
chmod +x deploy.sh
./deploy.sh

# Configure environment
cp env.example .env
nano .env  # Edit with your production values
```

#### 3. Nginx Configuration
```bash
# Install nginx configuration
sudo cp nginx.conf /etc/nginx/sites-available/neuralink-bci
sudo ln -s /etc/nginx/sites-available/neuralink-bci /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

#### 4. Systemd Service
```bash
# Install service
sudo cp neuralink-bci.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable neuralink-bci
sudo systemctl start neuralink-bci

# Check status
sudo systemctl status neuralink-bci
```

### SSL Certificate (Let's Encrypt)
```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📊 Monitoring and Maintenance

### Health Checks
```bash
# Check service health
./health_check.sh

# Monitor logs
tail -f logs/app.log
tail -f logs/access.log
tail -f logs/error.log

# System status
sudo systemctl status neuralink-bci
sudo systemctl status nginx
```

### Performance Monitoring
```bash
# System resources
htop
df -h
free -h

# Application metrics
curl http://localhost:5000/api/status
curl http://localhost:5000/api/health
```

### Backup Strategy
```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/neuralink-bci-$DATE"

mkdir -p $BACKUP_DIR
cp -r models/ $BACKUP_DIR/
cp -r logs/ $BACKUP_DIR/
cp .env $BACKUP_DIR/

# Compress backup
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "Backup created: $BACKUP_DIR.tar.gz"
EOF

chmod +x backup.sh

# Schedule backups
crontab -e
# Add: 0 2 * * * /path/to/backup.sh
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port 5000
sudo lsof -i :5000

# Kill process
sudo kill -9 PID
```

#### 2. Permission Issues
```bash
# Fix file permissions
sudo chown -R www-data:www-data /path/to/NeuroLink-BCI
sudo chmod -R 755 /path/to/NeuroLink-BCI
```

#### 3. Model Not Found
```bash
# Train model
python scripts/quick_train_model.py

# Check model path
ls -la models/
```

#### 4. Frontend Build Issues
```bash
# Clear cache and rebuild
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Log Analysis
```bash
# Application logs
grep "ERROR" logs/app.log
grep "WARNING" logs/app.log

# Nginx logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log

# System logs
sudo journalctl -u neuralink-bci -f
```

## 🚀 Scaling

### Horizontal Scaling
- Use load balancer (nginx, HAProxy)
- Deploy multiple backend instances
- Use Redis for session storage
- Implement database clustering

### Vertical Scaling
- Increase server resources
- Optimize model inference
- Use GPU acceleration
- Implement caching strategies

## 📈 Performance Optimization

### Backend Optimization
```python
# Enable gzip compression
from flask_compress import Compress
Compress(app)

# Use Redis for caching
import redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Optimize model loading
model = torch.jit.load('model.pt')  # TorchScript optimization
```

### Frontend Optimization
```javascript
// Enable compression
const compression = require('compression');
app.use(compression());

// Implement lazy loading
const LazyComponent = React.lazy(() => import('./Component'));

// Use service workers for caching
// Add to public/sw.js
```

## 🔒 Security Considerations

### Production Security
- Use HTTPS with valid SSL certificates
- Implement rate limiting
- Add authentication/authorization
- Regular security updates
- Firewall configuration
- Input validation and sanitization

### Environment Security
```bash
# Secure .env file
chmod 600 .env

# Use secrets management
# AWS Secrets Manager, Azure Key Vault, etc.

# Regular security audits
npm audit
pip-audit
```

## 📞 Support

For deployment issues:
1. Check logs for error messages
2. Verify configuration files
3. Test individual components
4. Review system resources
5. Contact support team

---

**Happy Deploying! 🎉**
