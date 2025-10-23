#!/bin/bash
# GitHub Repository Setup Script for NeuroLink-BCI

set -e

echo " Setting up GitHub repository for NeuroLink-BCI..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_status "Initializing Git repository..."
    git init
fi

# Add all files to git
print_status "Adding files to Git..."
git add .

# Create initial commit
print_status "Creating initial commit..."
git commit -m "Initial commit: NeuroLink-BCI Real-Time Neural Decoding System

- Complete CNN-LSTM hybrid model for EEG classification
- Real-time dashboard with interactive visualizations
- Comprehensive training analysis and metrics
- Production-ready deployment configuration
- Docker support for easy deployment
- CI/CD pipeline with GitHub Actions"

# Create main branch
print_status "Setting up main branch..."
git branch -M main

# Create develop branch
print_status "Creating develop branch..."
git checkout -b develop

# Switch back to main
git checkout main

print_header "Repository setup completed!"
echo ""
echo " Next steps:"
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Name it 'NeuroLink-BCI'"
echo "   - Don't initialize with README (we already have one)"
echo ""
echo "2. Add the remote origin:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/NeuroLink-BCI.git"
echo ""
echo "3. Push to GitHub:"
echo "   git push -u origin main"
echo "   git push -u origin develop"
echo ""
echo "4. Set up branch protection rules:"
echo "   - Go to Settings > Branches"
echo "   - Add rule for 'main' branch"
echo "   - Require pull request reviews"
echo "   - Require status checks to pass"
echo ""
echo "5. Enable GitHub Actions:"
echo "   - Go to Actions tab"
echo "   - Enable workflows"
echo ""
echo "6. Set up deployment:"
echo "   - Configure your production server"
echo "   - Run ./deploy.sh for production deployment"
echo ""
echo " Your NeuroLink-BCI project is ready for GitHub!"

# Create a simple push script
cat > push_to_github.sh << 'EOF'
#!/bin/bash
# Quick push script for NeuroLink-BCI

echo " Pushing NeuroLink-BCI to GitHub..."

# Check if remote exists
if ! git remote get-url origin &> /dev/null; then
    echo "Please add your GitHub remote first:"
    echo "git remote add origin https://github.com/YOUR_USERNAME/NeuroLink-BCI.git"
    exit 1
fi

# Add all changes
git add .

# Commit with timestamp
git commit -m "Update: $(date '+%Y-%m-%d %H:%M:%S')"

# Push to both branches
git push origin main
git push origin develop

echo " Successfully pushed to GitHub!"
EOF

chmod +x push_to_github.sh

print_status "Created push_to_github.sh script for easy updates"
