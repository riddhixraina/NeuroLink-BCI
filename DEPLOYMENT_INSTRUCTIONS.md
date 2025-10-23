# 🚀 GitHub Repository Setup & Live Deployment Guide

## Step 1: Create GitHub Repository

### Option A: Using GitHub Web Interface (Recommended)
1. **Go to GitHub**: https://github.com/new
2. **Repository Name**: `NeuroLink-BCI`
3. **Description**: `Real-Time Neural Decoding System with CNN-LSTM Architecture`
4. **Visibility**: Public (for portfolio showcase)
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. **Click "Create repository"**

### Option B: Using GitHub CLI (if installed)
```bash
gh repo create NeuroLink-BCI --public --description "Real-Time Neural Decoding System with CNN-LSTM Architecture"
```

## Step 2: Connect Local Repository to GitHub

After creating the repository on GitHub, run these commands:

```bash
# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/NeuroLink-BCI.git

# Push to GitHub
git push -u origin main
```

## Step 3: Deploy to Live Website

### Option A: Vercel (Recommended for Frontend)
1. **Go to**: https://vercel.com
2. **Sign up/Login** with GitHub
3. **Import Project**: Select your NeuroLink-BCI repository
4. **Framework Preset**: React
5. **Build Command**: `cd frontend && npm run build`
6. **Output Directory**: `frontend/build`
7. **Deploy**

### Option B: Netlify (Alternative)
1. **Go to**: https://netlify.com
2. **Sign up/Login** with GitHub
3. **New site from Git**: Select NeuroLink-BCI
4. **Build settings**:
   - Build command: `cd frontend && npm run build`
   - Publish directory: `frontend/build`
5. **Deploy**

### Option C: Railway (Full Stack)
1. **Go to**: https://railway.app
2. **Sign up/Login** with GitHub
3. **New Project**: From GitHub repo
4. **Select**: NeuroLink-BCI
5. **Deploy**

## Step 4: Backend Deployment

### Option A: Railway (Recommended)
1. **Create new service** in Railway
2. **Connect GitHub repo**
3. **Environment variables**:
   ```
   FLASK_ENV=production
   SECRET_KEY=your-secret-key
   ```
4. **Deploy**

### Option B: Heroku
1. **Install Heroku CLI**
2. **Login**: `heroku login`
3. **Create app**: `heroku create neuralink-bci`
4. **Set config**: 
   ```bash
   heroku config:set FLASK_ENV=production
   heroku config:set SECRET_KEY=your-secret-key
   ```
5. **Deploy**: `git push heroku main`

### Option C: Render
1. **Go to**: https://render.com
2. **New Web Service**
3. **Connect GitHub**: Select NeuroLink-BCI
4. **Build Command**: `pip install -r backend/requirements-prod.txt`
5. **Start Command**: `gunicorn --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker --bind 0.0.0.0:$PORT backend.app:app`
6. **Deploy**

## Step 5: Update Frontend Configuration

After deploying backend, update frontend to use live API:

```javascript
// In frontend/src/App.js, update API_BASE_URL
const API_BASE_URL = 'https://your-backend-url.com';
```

## Step 6: Custom Domain (Optional)

### For Vercel/Netlify:
1. **Go to project settings**
2. **Domains section**
3. **Add custom domain**
4. **Configure DNS** as instructed

## Quick Commands Summary

```bash
# 1. Create GitHub repo (manual step)
# 2. Connect and push
git remote add origin https://github.com/YOUR_USERNAME/NeuroLink-BCI.git
git push -u origin main

# 3. Deploy frontend to Vercel (manual step)
# 4. Deploy backend to Railway (manual step)
# 5. Update frontend API URL
# 6. Redeploy frontend
```

## Expected Results

After deployment, you'll have:
- **GitHub Repository**: https://github.com/YOUR_USERNAME/NeuroLink-BCI
- **Live Frontend**: https://your-app.vercel.app
- **Live Backend**: https://your-app.railway.app
- **Complete Documentation**: Professional README and docs

## Troubleshooting

### Common Issues:
1. **Build fails**: Check Node.js version (16+)
2. **API not connecting**: Verify CORS settings
3. **Model not loading**: Check file paths in production
4. **WebSocket issues**: Ensure production WebSocket support

### Support:
- Check deployment logs
- Verify environment variables
- Test API endpoints manually
- Check browser console for errors
