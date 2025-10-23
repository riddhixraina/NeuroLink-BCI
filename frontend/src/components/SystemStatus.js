import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Grid,
  LinearProgress
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Warning,
  Info
} from '@mui/icons-material';

const SystemStatus = ({ status }) => {
  const getStatusIcon = (status) => {
    switch (status) {
      case 'running':
        return <CheckCircle color="success" />;
      case 'error':
        return <Error color="error" />;
      case 'warning':
        return <Warning color="warning" />;
      default:
        return <Info color="info" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running':
        return 'success';
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      default:
        return 'info';
    }
  };

  const getComponentStatus = (component) => {
    return component ? 'Loaded' : 'Not Loaded';
  };

  const getComponentColor = (component) => {
    return component ? 'success' : 'error';
  };

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          {getStatusIcon(status?.status)}
          <Typography variant="h6" sx={{ ml: 1, fontWeight: 'bold' }}>
            System Status
          </Typography>
          <Chip
            label={status?.status?.toUpperCase() || 'UNKNOWN'}
            color={getStatusColor(status?.status)}
            size="small"
            sx={{ ml: 2 }}
          />
        </Box>

        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Backend Status
              </Typography>
              <Chip
                label={status?.status === 'running' ? 'Online' : 'Offline'}
                color={status?.status === 'running' ? 'success' : 'error'}
                size="small"
              />
            </Box>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Streaming Status
              </Typography>
              <Chip
                label={status?.streaming_active ? 'Active' : 'Inactive'}
                color={status?.streaming_active ? 'primary' : 'default'}
                size="small"
              />
            </Box>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Components Loaded
              </Typography>
              <Chip
                label={status?.components_loaded ? 'All Loaded' : 'Partial'}
                color={status?.components_loaded ? 'success' : 'warning'}
                size="small"
              />
            </Box>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Last Update
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {status?.timestamp ? 
                  new Date(status.timestamp).toLocaleTimeString() : 
                  'Unknown'}
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {status && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Component Status:
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={6} sm={3}>
                <Chip
                  label={`Data Loader: ${getComponentStatus(status.components_loaded)}`}
                  color={getComponentColor(status.components_loaded)}
                  size="small"
                  variant="outlined"
                />
              </Grid>
              <Grid item xs={6} sm={3}>
                <Chip
                  label={`Preprocessor: ${getComponentStatus(status.components_loaded)}`}
                  color={getComponentColor(status.components_loaded)}
                  size="small"
                  variant="outlined"
                />
              </Grid>
              <Grid item xs={6} sm={3}>
                <Chip
                  label={`Feature Extractor: ${getComponentStatus(status.components_loaded)}`}
                  color={getComponentColor(status.components_loaded)}
                  size="small"
                  variant="outlined"
                />
              </Grid>
              <Grid item xs={6} sm={3}>
                <Chip
                  label={`Classifier: ${getComponentStatus(status.components_loaded)}`}
                  color={getComponentColor(status.components_loaded)}
                  size="small"
                  variant="outlined"
                />
              </Grid>
            </Grid>
          </Box>
        )}

        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            System Health:
          </Typography>
          <LinearProgress
            variant="determinate"
            value={status?.components_loaded ? 100 : 50}
            color={status?.components_loaded ? 'success' : 'warning'}
            sx={{ height: 6, borderRadius: 3 }}
          />
        </Box>
      </CardContent>
    </Card>
  );
};

export default SystemStatus;
