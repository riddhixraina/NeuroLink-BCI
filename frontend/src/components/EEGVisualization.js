import React, { useState, useEffect, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { Box, Typography } from '@mui/material';

const EEGVisualization = ({ eegData, streaming }) => {
  const [plotData, setPlotData] = useState([]);
  const [layout, setLayout] = useState({});
  const plotRef = useRef(null);
  const dataBufferRef = useRef([]);
  const lastUpdateRef = useRef(0);

  const updatePlot = useCallback((data) => {
    const now = Date.now();
    
    // Throttle updates to prevent flickering
    if (now - lastUpdateRef.current < 200) { // Update max every 200ms
      return;
    }
    lastUpdateRef.current = now;
    
    console.log('EEGVisualization: updatePlot called with data:', data);
    const { eeg_data, channel_names, sampling_rate } = data;
    
    if (!eeg_data || !channel_names) {
      console.log('EEGVisualization: Missing data - eeg_data:', !!eeg_data, 'channel_names:', !!channel_names);
      return;
    }

    console.log('EEGVisualization: Processing data - channels:', eeg_data.length, 'samples:', eeg_data[0]?.length);

    // Create time axis (assuming 2 seconds of data)
    const timePoints = eeg_data[0].length;
    const duration = timePoints / sampling_rate;
    const timeAxis = Array.from({ length: timePoints }, (_, i) => i / sampling_rate);

    // Prepare plot data for first 8 channels (for better visualization)
    const maxChannels = Math.min(8, channel_names.length);
    const traces = [];

    for (let i = 0; i < maxChannels; i++) {
      const offset = i * 100; // Offset for better visualization
      const yValues = eeg_data[i].map(value => value + offset);
      
      console.log(`EEGVisualization: Channel ${i} - first 5 values:`, yValues.slice(0, 5));
      
      traces.push({
        x: timeAxis,
        y: yValues,
        type: 'scatter',
        mode: 'lines',
        name: channel_names[i],
        line: {
          width: 1,
          color: getChannelColor(i)
        },
        showlegend: false
      });
    }

    console.log('EEGVisualization: Setting plot data with', traces.length, 'traces');
    setPlotData(traces);

    setLayout({
      title: {
        text: 'Real-Time EEG Signals',
        font: { size: 16 }
      },
      xaxis: {
        title: 'Time (s)',
        range: [0, duration],
        showgrid: true,
        gridcolor: '#f0f0f0'
      },
      yaxis: {
        title: 'Amplitude (μV)',
        showgrid: true,
        gridcolor: '#f0f0f0',
        tickmode: 'array',
        tickvals: Array.from({ length: maxChannels }, (_, i) => i * 100),
        ticktext: channel_names.slice(0, maxChannels)
      },
      margin: { l: 60, r: 20, t: 40, b: 40 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { size: 12 }
    });
  }, []);

  useEffect(() => {
    console.log('EEGVisualization: useEffect triggered - eegData:', eegData);
    if (eegData && eegData.eeg_data && eegData.channel_names) {
      console.log('EEGVisualization: Calling updatePlot with eegData');
      updatePlot(eegData);
    } else {
      console.log('EEGVisualization: No valid eegData to plot');
    }
  }, [eegData, updatePlot]);

  const getChannelColor = (index) => {
    const colors = [
      '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ];
    return colors[index % colors.length];
  };

  return (
    <Box sx={{ height: '100%', width: '100%' }}>
      <Plot
        ref={plotRef}
        data={plotData}
        layout={layout}
        style={{ width: '100%', height: '100%' }}
        config={{
          responsive: true,
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
          mapboxAccessToken: null,
          toImageButtonOptions: {
            format: 'png',
            filename: 'eeg_plot',
            height: 500,
            width: 800,
            scale: 1
          }
        }}
      />
      
      {!streaming && (
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
            color: 'text.secondary'
          }}
        >
          <Typography variant="h6">
            {eegData ? 'Streaming Stopped' : 'Waiting for EEG Data...'}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default EEGVisualization;
