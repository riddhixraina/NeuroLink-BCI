// Production configuration for NeuroLink-BCI Frontend
const config = {
  // API Configuration
  API_BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:5000',
  
  // WebSocket Configuration  
  WS_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:5000',
  
  // Environment
  NODE_ENV: process.env.NODE_ENV || 'development',
  
  // Feature flags
  ENABLE_ANALYTICS: process.env.REACT_APP_ENABLE_ANALYTICS === 'true',
  ENABLE_DEBUG: process.env.REACT_APP_DEBUG === 'true',
  
  // Performance settings
  REFRESH_INTERVAL: parseInt(process.env.REACT_APP_REFRESH_INTERVAL) || 1000,
  MAX_RETRIES: parseInt(process.env.REACT_APP_MAX_RETRIES) || 3,
  
  // UI Configuration
  THEME: process.env.REACT_APP_THEME || 'light',
  LANGUAGE: process.env.REACT_APP_LANGUAGE || 'en',
};

export default config;
