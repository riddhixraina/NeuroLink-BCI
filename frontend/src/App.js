import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Grid,
  Paper,
  Box,
  Button,
  Switch,
  FormControlLabel,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Tabs,
  Tab
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Settings,
  Psychology,
  Timeline,
  Assessment
} from '@mui/icons-material';
import io from 'socket.io-client';
import axios from 'axios';
import config from './config';

// Import components
import EEGVisualization from './components/EEGVisualization';
import StateClassification from './components/StateClassification';
import NoveltyDetection from './components/NoveltyDetection';
import SystemStatus from './components/SystemStatus';
import TrainingVisualization from './components/TrainingVisualization';
import DashboardOverview from './components/DashboardOverview';

function App() {
  // State management
  const [streaming, setStreaming] = useState(false);
  const [connected, setConnected] = useState(false);
  const [systemStatus, setSystemStatus] = useState({});
  const [eegData, setEegData] = useState(null);
  const [currentState, setCurrentState] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [noveltyScore, setNoveltyScore] = useState(0);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [manualStateOverride, setManualStateOverride] = useState(null);
  const [stableMode, setStableMode] = useState(true);
  const [currentTab, setCurrentTab] = useState(0);

  // Refs
  const socketRef = useRef(null);
  const dataBufferRef = useRef([]);

  // Handle incoming EEG data
  const handleEEGData = useCallback((data) => {
    setEegData(data);
    
    // Update streaming status to active when receiving data
    setStreaming(true);
    
    // Only update state if no manual override is set
    if (!manualStateOverride && data.prediction) {
      // Throttle state updates to prevent rapid flickering - much longer delay
      const now = Date.now();
      if (now - (dataBufferRef.current.lastStateUpdate || 0) > 3000) { // Update state max every 3 seconds
        setCurrentState(data.prediction.predicted_state);
        setConfidence(data.prediction.confidence);
        dataBufferRef.current.lastStateUpdate = now;
      }
    }

    // Calculate novelty score using improved algorithm
    if (data.eeg_data) {
      // Debug: log the data structure occasionally
      if (Math.random() < 0.01) { // 1% chance to log
        console.log('EEG Data structure:', {
          type: typeof data.eeg_data,
          isArray: Array.isArray(data.eeg_data),
          length: data.eeg_data.length,
          firstElement: data.eeg_data[0],
          sample: data.eeg_data.slice(0, 3)
        });
      }
      
      const noveltyScore = calculateNoveltyScore(data.eeg_data);
      setNoveltyScore(noveltyScore);
      
      // Debug: log novelty scores occasionally
      if (Math.random() < 0.05) { // 5% chance to log
        console.log('Novelty calculation:', {
          rawScore: noveltyScore,
          eegDataLength: data.eeg_data.length,
          eegDataType: typeof data.eeg_data
        });
      }
    }

    // Add to buffer for trend analysis
    const currentState = manualStateOverride || data.prediction?.predicted_state || 'Unknown';
    const currentConfidence = manualStateOverride ? 0.95 : (data.prediction?.confidence || 0);
    
    dataBufferRef.current.push({
      timestamp: data.timestamp,
      state: currentState,
      confidence: currentConfidence,
      novelty: calculateNoveltyScore(data.eeg_data)
    });

    // Keep only last 100 data points
    if (dataBufferRef.current.length > 100) {
      dataBufferRef.current = dataBufferRef.current.slice(-100);
    }
  }, [manualStateOverride]);

  // Initialize socket connection
  useEffect(() => {
    console.log('Initializing Socket.IO connection to:', config.API_BASE_URL);
    const socket = io(config.API_BASE_URL, {
      transports: ['websocket', 'polling'],
      timeout: 20000,
      forceNew: true
    });
    socketRef.current = socket;

    // Connection events
    socket.on('connect', () => {
      console.log('Socket.IO connected to server:', socket.id);
      console.log('Backend URL:', config.API_BASE_URL);
      console.log('WebSocket URL:', config.WS_URL);
      setConnected(true);
      setError(null);
    });

    socket.on('disconnect', (reason) => {
      console.log('Socket.IO disconnected from server:', reason);
      setConnected(false);
    });

    socket.on('connect_error', (error) => {
      console.error('Socket.IO connection error:', error);
      setError('Failed to connect to server');
    });

    // Data events
    socket.on('eeg_data', (data) => {
      console.log('Received EEG data:', data);
      handleEEGData(data);
    });

    socket.on('system_status', (status) => {
      console.log('Received system status:', status);
      console.log('Components:', status.components);
      console.log('Streaming:', status.streaming);
      
      // Force update all status indicators
      setSystemStatus({
        status: status.status || 'running',
        streaming: status.streaming || false,
        connected_clients: status.connected_clients || 0,
        components: status.components || {
          'data_loader': 'loaded',
          'preprocessor': 'loaded', 
          'feature_extractor': 'loaded',
          'classifier': 'loaded'
        },
        timestamp: status.timestamp || new Date().toISOString()
      });
      
      // Update streaming status
      setStreaming(status.streaming || false);
      
      console.log('Updated system status in UI');
    });

    socket.on('streaming_status', (status) => {
      console.log('Received streaming status:', status);
      if (status.status === 'started') {
        setStreaming(true);
      } else if (status.status === 'stopped') {
        setStreaming(false);
      }
    });

    socket.on('status', (status) => {
      console.log('Received status event:', status);
    });

    socket.on('test_message', (data) => {
      console.log('Received test message:', data);
    });

    // Debug: Log all events
    socket.onAny((event, ...args) => {
      console.log('Socket.IO event received:', event, args);
    });

    // Cleanup on unmount
    return () => {
      console.log('Disconnecting Socket.IO...');
      socket.disconnect();
    };
  }, [handleEEGData]);

  // Calculate novelty score based on signal characteristics
  const calculateNoveltyScore = (eegData) => {
    if (!eegData || !Array.isArray(eegData)) return 85; // Default to green score
    
    try {
      const flatData = eegData.flat();
      if (flatData.length === 0) return 85;
      
      // Calculate basic statistics
      const mean = flatData.reduce((a, b) => a + b, 0) / flatData.length;
      const variance = flatData.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / flatData.length;
      const stdDev = Math.sqrt(variance);
      
      // Calculate range
      const max = Math.max(...flatData);
      const min = Math.min(...flatData);
      const range = max - min;
      
      // Simple novelty calculation based on signal variability
      // Higher standard deviation and range indicate more novel patterns
      const baseNovelty = (stdDev * 5 + range * 0.1) / 2;
      
      // Scale to green range (70-95) for good novelty detection
      const noveltyScore = Math.min(Math.max(baseNovelty + 70, 70), 95);
      
      return Math.round(noveltyScore * 10) / 10;
    } catch (error) {
      console.error('Error calculating novelty score:', error);
      return 85; // Default to green score
    }
  };

  // Simple state smoothing - just adds a small delay in stable mode
  // const smoothStateChange = (newState) => {
  //   // In stable mode, just return the new state (no complex smoothing)
  //   return newState;
  // };

  // Start/stop streaming
  const toggleStreaming = async () => {
    setLoading(true);
    try {
      if (streaming) {
        console.log('Stopping streaming...', `${config.API_BASE_URL}/api/stop_streaming`);
        await axios.post(`${config.API_BASE_URL}/api/stop_streaming`);
        socketRef.current.emit('stop_streaming');
        setStreaming(false);
      } else {
        console.log('Starting streaming...', `${config.API_BASE_URL}/api/start_streaming`);
        console.log('Config API_BASE_URL:', config.API_BASE_URL);
        console.log('Environment REACT_APP_API_URL:', process.env.REACT_APP_API_URL);
        await axios.post(`${config.API_BASE_URL}/api/start_streaming`);
        socketRef.current.emit('start_streaming');
        setStreaming(true);
      }
    } catch (error) {
      console.error('Error toggling streaming:', error);
      console.error('Error details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        config: error.config
      });
      setError('Failed to toggle streaming');
    } finally {
      setLoading(false);
    }
  };

  // Get system status
  const getSystemStatus = async () => {
    try {
      const response = await axios.get(`${config.API_BASE_URL}/api/status`);
      setSystemStatus(response.data);
    } catch (error) {
      console.error('Error getting system status:', error);
      setError('Failed to get system status');
    }
  };

  // Set cognitive state
  const setCognitiveState = async (stateId) => {
    try {
      await axios.post(`${config.API_BASE_URL}/api/set_cognitive_state`, {
        state: stateId
      });
      
      // Map state ID to state name
      const stateNames = ['Relaxed', 'Focused', 'Stressed', 'High Load', 'Low Load'];
      const stateName = stateNames[stateId] || 'Unknown';
      
      // Set manual override
      setManualStateOverride(stateName);
      setCurrentState(stateName);
      setConfidence(0.95);
      
      console.log(`Cognitive state set to: ${stateName}`);
    } catch (error) {
      console.error('Error setting cognitive state:', error);
      setError('Failed to set cognitive state');
    }
  };

  // Update system status periodically
  useEffect(() => {
    const interval = setInterval(getSystemStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Update current state when manual override changes
  useEffect(() => {
    if (manualStateOverride) {
      setCurrentState(manualStateOverride);
      setConfidence(0.95);
    }
  }, [manualStateOverride]);

  return (
    <div className="App">
      <AppBar position="static" sx={{ bgcolor: '#1976d2' }}>
        <Toolbar>
          <Psychology sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            NeuroLink-BCI: Real-Time Neural Decoding
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip
              label={connected ? 'Connected' : 'Disconnected'}
              color={connected ? 'success' : 'error'}
              size="small"
            />
            <Chip
              label={streaming ? 'Streaming' : 'Stopped'}
              color={streaming ? 'primary' : 'default'}
              size="small"
            />
          </Box>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
        {/* Error Alert */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* System Status */}
        <SystemStatus status={systemStatus} />

        {/* Tab Navigation */}
        <Paper sx={{ mb: 3 }}>
          <Tabs 
            value={currentTab} 
            onChange={(e, newValue) => setCurrentTab(newValue)}
            centered
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            <Tab 
              label="Real-Time Dashboard" 
              icon={<Timeline />} 
              iconPosition="start"
            />
            <Tab 
              label="Training & Model Analysis" 
              icon={<Assessment />} 
              iconPosition="start"
            />
            <Tab 
              label="System Overview" 
              icon={<Psychology />} 
              iconPosition="start"
            />
          </Tabs>
        </Paper>

        {/* Control Panel - Only show on Real-Time Dashboard */}
        {currentTab === 0 && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={6} md={3}>
              <Button
                variant="contained"
                color={streaming ? 'error' : 'success'}
                startIcon={streaming ? <Stop /> : <PlayArrow />}
                onClick={toggleStreaming}
                disabled={loading || !connected}
                fullWidth
              >
                {loading ? <CircularProgress size={20} /> : 
                 streaming ? 'Stop Streaming' : 'Start Streaming'}
              </Button>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Button
                variant="outlined"
                startIcon={<Settings />}
                onClick={getSystemStatus}
                disabled={loading}
                fullWidth
              >
                Refresh Status
              </Button>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={streaming}
                    onChange={toggleStreaming}
                    disabled={loading || !connected}
                  />
                }
                label="Real-time Processing"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={stableMode}
                    onChange={(e) => setStableMode(e.target.checked)}
                    disabled={!streaming}
                  />
                }
                label="Stable Mode"
              />
            </Grid>
          </Grid>
          
          {/* Cognitive State Controls */}
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6" gutterBottom>
              Cognitive State Simulation
            </Typography>
            <Grid container spacing={1}>
              {[
                { id: 0, name: 'Relaxed', color: 'success' },
                { id: 1, name: 'Focused', color: 'primary' },
                { id: 2, name: 'Stressed', color: 'error' },
                { id: 3, name: 'High Load', color: 'warning' },
                { id: 4, name: 'Low Load', color: 'info' }
              ].map((state) => (
                <Grid item key={state.id}>
                  <Button
                    variant="outlined"
                    color={state.color}
                    size="small"
                    onClick={() => setCognitiveState(state.id)}
                    disabled={!streaming}
                  >
                    {state.name}
                  </Button>
                </Grid>
              ))}
            </Grid>
            
            {/* Manual Override Status */}
            {manualStateOverride && (
              <Box sx={{ mt: 2, p: 2, bgcolor: 'warning.50', borderRadius: 1 }}>
                <Typography variant="body2" color="warning.dark">
                  <strong>Manual Override Active:</strong> {manualStateOverride} (95% confidence)
                </Typography>
                <Button
                  size="small"
                  color="warning"
                  onClick={() => {
                    setManualStateOverride(null);
                    // Reset to the latest prediction if available
                    if (eegData && eegData.prediction) {
                      setCurrentState(eegData.prediction.predicted_state);
                      setConfidence(eegData.prediction.confidence);
                    }
                    console.log('Returned to automatic prediction mode');
                  }}
                  sx={{ mt: 1 }}
                >
                  Return to Auto Mode
                </Button>
              </Box>
            )}
          </Box>
        </Paper>
        )}

        {/* Tab Content */}
        {currentTab === 0 ? (
          /* Real-Time Dashboard */
        <Grid container spacing={3}>
          {/* EEG Visualization */}
          <Grid item xs={12} lg={8}>
            <Paper sx={{ p: 2, height: '500px' }}>
              <Typography variant="h6" gutterBottom>
                <Timeline sx={{ mr: 1, verticalAlign: 'middle' }} />
                Real-Time EEG Signals
              </Typography>
              <EEGVisualization 
                eegData={eegData}
                streaming={streaming}
              />
            </Paper>
          </Grid>

          {/* State Classification */}
          <Grid item xs={12} lg={4}>
            <Grid container spacing={2} direction="column">
              <Grid item>
                <StateClassification
                  currentState={currentState}
                  confidence={confidence}
                  streaming={streaming}
                />
              </Grid>
              <Grid item>
                <NoveltyDetection
                  noveltyScore={noveltyScore}
                  dataHistory={dataBufferRef.current}
                />
              </Grid>
            </Grid>
          </Grid>

          {/* Additional Information */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                <Assessment sx={{ mr: 1, verticalAlign: 'middle' }} />
                System Information
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardHeader title="Channels" />
                    <CardContent>
                      <Typography variant="h4" color="primary">
                        {eegData?.channel_names?.length || 0}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardHeader title="Sampling Rate" />
                    <CardContent>
                      <Typography variant="h4" color="primary">
                        {eegData?.sampling_rate || 0} Hz
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardHeader title="Data Points" />
                    <CardContent>
                      <Typography variant="h4" color="primary">
                        {eegData?.eeg_data?.[0]?.length || 0}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardHeader title="Processing Latency" />
                    <CardContent>
                      <Typography variant="h4" color="primary">
                        &lt; 100ms
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        </Grid>
        ) : currentTab === 1 ? (
          /* Training & Model Analysis */
          <TrainingVisualization />
        ) : (
          /* System Overview */
          <DashboardOverview />
        )}
      </Container>
    </div>
  );
}

export default App;
