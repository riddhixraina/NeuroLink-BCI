import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Grid,
  Box,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Paper
} from '@mui/material';
import {
  Psychology,
  Timeline,
  Assessment,
  Speed,
  Memory,
  Security,
  CloudSync,
  Science,
  School
} from '@mui/icons-material';

const DashboardOverview = () => {
  const features = [
    {
      icon: <Psychology />,
      title: "Real-Time EEG Classification",
      description: "CNN-LSTM hybrid model processes EEG signals in real-time to classify cognitive states"
    },
    {
      icon: <Timeline />,
      title: "Live Signal Visualization",
      description: "Interactive plots showing 32-channel EEG data with 128Hz sampling rate"
    },
    {
      icon: <Assessment />,
      title: "Comprehensive Training Analysis",
      description: "Detailed training metrics, cross-validation results, and model performance visualization"
    },
    {
      icon: <Speed />,
      title: "Low-Latency Processing",
      description: "Sub-100ms processing latency for real-time applications"
    },
    {
      icon: <Memory />,
      title: "Novelty Detection",
      description: "Advanced algorithms to detect unusual patterns in neural activity"
    },
    {
      icon: <Security />,
      title: "Robust Architecture",
      description: "Early stopping, cross-validation, and comprehensive evaluation protocols"
    }
  ];

  const cognitiveStates = [
    { id: 0, name: "Relaxed", color: "#4CAF50", description: "Calm, meditative state with high alpha waves" },
    { id: 1, name: "Focused", color: "#2196F3", description: "Concentrated attention with balanced alpha/beta activity" },
    { id: 2, name: "Stressed", color: "#F44336", description: "High arousal state with elevated beta/gamma waves" },
    { id: 3, name: "High Load", color: "#FF9800", description: "Cognitive overload with very high beta/gamma activity" },
    { id: 4, name: "Low Load", color: "#9C27B0", description: "Minimal cognitive activity with low overall power" }
  ];

  const technicalSpecs = [
    "32 EEG channels (standard 10-20 montage)",
    "128 Hz sampling rate",
    "256 time points per analysis window",
    "CNN-LSTM hybrid architecture",
    "5-fold cross-validation",
    "Early stopping with patience=15",
    "Adam optimizer with learning rate scheduling",
    "Real-time WebSocket streaming",
    "Flask backend with REST API",
    "React frontend with Material-UI"
  ];

  return (
    <Box sx={{ p: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', minHeight: '100vh' }}>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3, background: 'rgba(255, 255, 255, 0.95)', backdropFilter: 'blur(10px)' }}>
        <Box sx={{ textAlign: 'center', mb: 2 }}>
          <Psychology sx={{ fontSize: 60, color: '#1976d2', mb: 1 }} />
          <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold', color: '#1976d2' }}>
            NeuroLink-BCI
          </Typography>
          <Typography variant="h5" color="text.secondary" gutterBottom>
            Real-Time Neural Decoding System
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 800, mx: 'auto', mt: 2 }}>
            A comprehensive Brain-Computer Interface system that uses advanced machine learning to decode 
            cognitive states from EEG signals in real-time. This prototype demonstrates state-of-the-art 
            neural signal processing and classification capabilities.
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 3 }}>
          <Chip icon={<Science />} label="Machine Learning" color="primary" />
          <Chip icon={<CloudSync />} label="Real-Time Processing" color="secondary" />
          <Chip icon={<School />} label="Research Prototype" color="success" />
        </Box>
      </Paper>

      {/* System Overview */}
      <Grid container spacing={3}>
        {/* Features */}
        <Grid item xs={12} md={8}>
          <Card sx={{ height: '100%', background: 'rgba(255, 255, 255, 0.95)', backdropFilter: 'blur(10px)' }}>
            <CardHeader 
              title="System Features" 
              avatar={<Assessment sx={{ color: '#1976d2' }} />}
            />
            <CardContent>
              <Grid container spacing={2}>
                {features.map((feature, index) => (
                  <Grid item xs={12} sm={6} key={index}>
                    <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 2, height: '100%' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Box sx={{ color: '#1976d2', mr: 1 }}>
                          {feature.icon}
                        </Box>
                        <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                          {feature.title}
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {feature.description}
                      </Typography>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Cognitive States */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', background: 'rgba(255, 255, 255, 0.95)', backdropFilter: 'blur(10px)' }}>
            <CardHeader 
              title="Cognitive States" 
              avatar={<Psychology sx={{ color: '#1976d2' }} />}
            />
            <CardContent>
              <List>
                {cognitiveStates.map((state, index) => (
                  <React.Fragment key={state.id}>
                    <ListItem sx={{ px: 0 }}>
                      <ListItemIcon>
                        <Box 
                          sx={{ 
                            width: 20, 
                            height: 20, 
                            borderRadius: '50%', 
                            backgroundColor: state.color 
                          }} 
                        />
                      </ListItemIcon>
                      <ListItemText
                        primary={state.name}
                        secondary={state.description}
                        primaryTypographyProps={{ fontWeight: 'bold' }}
                        secondaryTypographyProps={{ fontSize: '0.8rem' }}
                      />
                    </ListItem>
                    {index < cognitiveStates.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Technical Specifications */}
        <Grid item xs={12}>
          <Card sx={{ background: 'rgba(255, 255, 255, 0.95)', backdropFilter: 'blur(10px)' }}>
            <CardHeader 
              title="Technical Specifications" 
              avatar={<Memory sx={{ color: '#1976d2' }} />}
            />
            <CardContent>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom sx={{ color: '#1976d2' }}>
                    Hardware Requirements
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemIcon><Speed /></ListItemIcon>
                      <ListItemText primary="32-channel EEG cap" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><Speed /></ListItemIcon>
                      <ListItemText primary="128 Hz sampling rate minimum" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><Speed /></ListItemIcon>
                      <ListItemText primary="Standard 10-20 electrode montage" />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><Speed /></ListItemIcon>
                      <ListItemText primary="Low-noise amplifiers" />
                    </ListItem>
                  </List>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom sx={{ color: '#1976d2' }}>
                    Software Architecture
                  </Typography>
                  <List dense>
                    {technicalSpecs.map((spec, index) => (
                      <ListItem key={index}>
                        <ListItemIcon><Security /></ListItemIcon>
                        <ListItemText primary={spec} />
                      </ListItem>
                    ))}
                  </List>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Metrics */}
        <Grid item xs={12}>
          <Card sx={{ background: 'rgba(255, 255, 255, 0.95)', backdropFilter: 'blur(10px)' }}>
            <CardHeader 
              title="Performance Metrics" 
              avatar={<Timeline sx={{ color: '#1976d2' }} />}
            />
            <CardContent>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h3" sx={{ color: '#4CAF50', fontWeight: 'bold' }}>
                      89%
                    </Typography>
                    <Typography variant="h6" gutterBottom>
                      Test Accuracy
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      On synthetic EEG data
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h3" sx={{ color: '#2196F3', fontWeight: 'bold' }}>
                      &lt;100ms
                    </Typography>
                    <Typography variant="h6" gutterBottom>
                      Processing Latency
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Real-time performance
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h3" sx={{ color: '#FF9800', fontWeight: 'bold' }}>
                      5-fold
                    </Typography>
                    <Typography variant="h6" gutterBottom>
                      Cross-Validation
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Robust evaluation
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h3" sx={{ color: '#9C27B0', fontWeight: 'bold' }}>
                      5
                    </Typography>
                    <Typography variant="h6" gutterBottom>
                      Cognitive States
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Classification targets
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Research Applications */}
        <Grid item xs={12}>
          <Card sx={{ background: 'rgba(255, 255, 255, 0.95)', backdropFilter: 'blur(10px)' }}>
            <CardHeader 
              title="Research Applications" 
              avatar={<School sx={{ color: '#1976d2' }} />}
            />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Typography variant="h6" gutterBottom sx={{ color: '#1976d2' }}>
                    Clinical Research
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    • Cognitive load assessment<br/>
                    • Attention deficit studies<br/>
                    • Stress monitoring<br/>
                    • Mental fatigue detection
                  </Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="h6" gutterBottom sx={{ color: '#1976d2' }}>
                    Human-Computer Interaction
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    • Adaptive interfaces<br/>
                    • Brain-controlled systems<br/>
                    • Workload optimization<br/>
                    • User experience enhancement
                  </Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="h6" gutterBottom sx={{ color: '#1976d2' }}>
                    Educational Technology
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    • Learning state monitoring<br/>
                    • Personalized education<br/>
                    • Attention tracking<br/>
                    • Cognitive assessment tools
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DashboardOverview;
