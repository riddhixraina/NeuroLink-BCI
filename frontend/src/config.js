// Production configuration for NeuroLink-BCI Frontend
const config = {
  // API Configuration - ensure it's a complete URL
  API_BASE_URL: (() => {
    const url = process.env.REACT_APP_API_URL || 'https://neurolink-bci-production.up.railway.app';
    // If URL doesn't start with http, add https://
    return url.startsWith('http') ? url : `https://${url}`;
  })(),
  
  // WebSocket Configuration - ensure it's a complete URL
  WS_URL: (() => {
    const url = process.env.REACT_APP_WS_URL || 'wss://neurolink-bci-production.up.railway.app';
    // If URL doesn't start with ws, add wss://
    return url.startsWith('ws') ? url : `wss://${url}`;
  })(),
  
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
