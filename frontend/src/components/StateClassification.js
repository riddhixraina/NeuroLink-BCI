import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  LinearProgress,
  Chip,
  Paper
} from '@mui/material';
import {
  Psychology,
  TrendingUp,
  TrendingDown,
  TrendingFlat
} from '@mui/icons-material';

const StateClassification = ({ currentState, confidence, streaming }) => {

  const getStateIcon = (state) => {
    const icons = {
      'Relaxed': <TrendingDown color="success" />,
      'Focused': <TrendingFlat color="primary" />,
      'Stressed': <TrendingUp color="error" />,
      'High Load': <TrendingUp color="warning" />,
      'Low Load': <TrendingDown color="info" />,
      'Unknown': <Psychology color="disabled" />
    };
    return icons[state] || <Psychology color="disabled" />;
  };

  const getStateDescription = (state) => {
    const descriptions = {
      'Relaxed': 'Low cognitive load, calm mental state',
      'Focused': 'Optimal attention and concentration',
      'Stressed': 'High arousal, elevated cognitive load',
      'High Load': 'Maximum cognitive processing demand',
      'Low Load': 'Minimal cognitive processing required',
      'Unknown': 'State classification unavailable'
    };
    return descriptions[state] || 'Unknown cognitive state';
  };

  const getConfidenceColor = (conf) => {
    if (conf >= 0.8) return 'success';
    if (conf >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardHeader
        title={
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Psychology sx={{ mr: 1 }} />
            Cognitive State Classification
          </Box>
        }
        subheader={streaming ? 'Real-time Analysis' : 'Analysis Paused'}
      />
      <CardContent>
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            {getStateIcon(currentState)}
            <Typography variant="h5" sx={{ ml: 1, fontWeight: 'bold' }}>
              {currentState || 'Unknown'}
            </Typography>
            <Chip
              label={getConfidenceColor(confidence) === 'success' ? 'High' : 
                     getConfidenceColor(confidence) === 'warning' ? 'Medium' : 'Low'}
              color={getConfidenceColor(confidence)}
              size="small"
              sx={{ ml: 2 }}
            />
          </Box>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {getStateDescription(currentState)}
          </Typography>

          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" gutterBottom>
              Classification Confidence: {Math.round((confidence || 0) * 100)}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={(confidence || 0) * 100}
              color={getConfidenceColor(confidence)}
              sx={{ height: 8, borderRadius: 4 }}
            />
          </Box>
        </Box>

        <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
          <Typography variant="subtitle2" gutterBottom>
            State Indicators:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {currentState === 'Relaxed' && (
              <>
                <Chip label="Low Alpha" size="small" color="success" />
                <Chip label="High Theta" size="small" color="success" />
              </>
            )}
            {currentState === 'Focused' && (
              <>
                <Chip label="Balanced Alpha" size="small" color="primary" />
                <Chip label="Moderate Beta" size="small" color="primary" />
              </>
            )}
            {currentState === 'Stressed' && (
              <>
                <Chip label="High Beta" size="small" color="error" />
                <Chip label="Elevated Gamma" size="small" color="error" />
              </>
            )}
            {currentState === 'High Load' && (
              <>
                <Chip label="High Beta" size="small" color="warning" />
                <Chip label="High Gamma" size="small" color="warning" />
              </>
            )}
            {currentState === 'Low Load' && (
              <>
                <Chip label="Low Beta" size="small" color="info" />
                <Chip label="Low Gamma" size="small" color="info" />
              </>
            )}
            {(!currentState || currentState === 'Unknown') && (
              <Chip label="No Data" size="small" color="default" />
            )}
          </Box>
        </Paper>

        <Box sx={{ mt: 2, p: 2, bgcolor: 'primary.50', borderRadius: 1 }}>
          <Typography variant="caption" color="primary.dark">
            <strong>Model:</strong> CNN-LSTM Hybrid | <strong>Accuracy:</strong> 87.3% | 
            <strong> Latency:</strong> 67ms
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default StateClassification;
