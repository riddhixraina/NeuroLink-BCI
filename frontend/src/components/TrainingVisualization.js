import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar
} from 'recharts';

const TrainingVisualization = () => {
  const [trainingData, setTrainingData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [trainingInProgress, setTrainingInProgress] = useState(false);

  // Colors for different states
  const stateColors = {
    0: '#4CAF50', // Relaxed - Green
    1: '#2196F3', // Focused - Blue
    2: '#F44336', // Stressed - Red
    3: '#FF9800', // High Load - Orange
    4: '#9C27B0'  // Low Load - Purple
  };

  const stateNames = {
    0: 'Relaxed',
    1: 'Focused',
    2: 'Stressed',
    3: 'High Load',
    4: 'Low Load'
  };

  useEffect(() => {
    fetchTrainingData();
    // Poll for training progress
    const interval = setInterval(fetchTrainingData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchTrainingData = async () => {
    try {
      const response = await fetch('/api/training/status');
      const data = await response.json();
      setTrainingData(data);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch training data');
      setLoading(false);
    }
  };

  const startTraining = async () => {
    try {
      setTrainingInProgress(true);
      const response = await fetch('/api/training/start', { method: 'POST' });
      const result = await response.json();
      
      if (result.status === 'training_started') {
        // Start polling for progress
        const progressInterval = setInterval(async () => {
          const progressResponse = await fetch('/api/training/progress');
          const progressData = await progressResponse.json();
          
          if (!progressData.training_in_progress) {
            clearInterval(progressInterval);
            setTrainingInProgress(false);
            fetchTrainingData(); // Refresh data
          }
        }, 2000);
      }
    } catch (err) {
      setError('Failed to start training');
      setTrainingInProgress(false);
    }
  };

  if (loading) {
    return (
      <div className="training-viz-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading training data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="training-viz-container">
        <div className="error-message">
          <h3>Error</h3>
          <p>{error}</p>
          <button onClick={fetchTrainingData}>Retry</button>
        </div>
      </div>
    );
  }

  if (!trainingData || !trainingData.model_exists) {
    return (
      <div className="training-viz-container">
        <div className="no-model-message">
          <h3>No Trained Model Found</h3>
          <p>Start training to see visualizations and metrics</p>
          <button 
            onClick={startTraining} 
            disabled={trainingInProgress}
            className="start-training-btn"
          >
            {trainingInProgress ? 'Training...' : 'Start Training'}
          </button>
        </div>
      </div>
    );
  }

  const { metrics, training_curves, test_results, cv_results, model_config, training_params } = trainingData;

  return (
    <div className="training-viz-container">
      <div className="training-header">
        <h2>🧠 Model Training Dashboard</h2>
        <div className="training-controls">
          <button 
            onClick={startTraining} 
            disabled={trainingInProgress}
            className="start-training-btn"
          >
            {trainingInProgress ? 'Training...' : 'Retrain Model'}
          </button>
          <button onClick={fetchTrainingData} className="refresh-btn">
            Refresh Data
          </button>
        </div>
      </div>

      {/* Model Overview */}
      <div className="model-overview">
        <h3>📊 Model Overview</h3>
        <div className="overview-grid">
          <div className="overview-card">
            <h4>Architecture</h4>
            <p>CNN-LSTM Hybrid</p>
            <p>Channels: {model_config.n_channels}</p>
            <p>Time Points: {model_config.n_timepoints}</p>
            <p>Classes: {model_config.n_classes}</p>
          </div>
          <div className="overview-card">
            <h4>Training Parameters</h4>
            <p>Batch Size: {training_params.batch_size}</p>
            <p>Epochs: {training_params.epochs}</p>
            <p>Learning Rate: {training_params.learning_rate}</p>
            <p>Early Stopping: {training_params.early_stopping_patience}</p>
          </div>
          <div className="overview-card">
            <h4>Performance</h4>
            <p>Test Accuracy: {(metrics?.test_accuracy * 100 || 0).toFixed(1)}%</p>
            <p>Best Val Loss: {metrics?.best_val_loss?.toFixed(4) || 'N/A'}</p>
            <p>Final Train Acc: {(metrics?.final_train_accuracy || 0).toFixed(1)}%</p>
            <p>Final Val Acc: {(metrics?.final_val_accuracy || 0).toFixed(1)}%</p>
          </div>
        </div>
      </div>

      {/* Training Curves */}
      {training_curves && (
        <div className="training-curves">
          <h3>📈 Training Progress</h3>
          <div className="charts-grid">
            <div className="chart-container">
              <h4>Loss Curves</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={training_curves.epochs.map((epoch, i) => ({
                  epoch,
                  'Training Loss': training_curves.train_losses[i],
                  'Validation Loss': training_curves.val_losses[i]
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="Training Loss" stroke="#2196F3" strokeWidth={2} />
                  <Line type="monotone" dataKey="Validation Loss" stroke="#F44336" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            <div className="chart-container">
              <h4>Accuracy Curves</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={training_curves.epochs.map((epoch, i) => ({
                  epoch,
                  'Training Accuracy': training_curves.train_accuracies[i],
                  'Validation Accuracy': training_curves.val_accuracies[i]
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="Training Accuracy" stroke="#4CAF50" strokeWidth={2} />
                  <Line type="monotone" dataKey="Validation Accuracy" stroke="#FF9800" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Performance Metrics */}
      {test_results && (
        <div className="performance-metrics">
          <h3>🎯 Performance Metrics</h3>
          <div className="metrics-grid">
            <div className="metric-card">
              <h4>Overall Performance</h4>
              <div className="metric-value">{(test_results.accuracy * 100).toFixed(1)}%</div>
              <div className="metric-label">Test Accuracy</div>
            </div>
            <div className="metric-card">
              <h4>Precision</h4>
              <div className="metric-value">{(test_results.precision * 100).toFixed(1)}%</div>
              <div className="metric-label">Weighted Average</div>
            </div>
            <div className="metric-card">
              <h4>Recall</h4>
              <div className="metric-value">{(test_results.recall * 100).toFixed(1)}%</div>
              <div className="metric-label">Weighted Average</div>
            </div>
            <div className="metric-card">
              <h4>F1-Score</h4>
              <div className="metric-value">{(test_results.f1_score * 100).toFixed(1)}%</div>
              <div className="metric-label">Weighted Average</div>
            </div>
          </div>
        </div>
      )}

      {/* Cross-Validation Results */}
      {cv_results && (
        <div className="cross-validation">
          <h3>🔄 Cross-Validation Results</h3>
          <div className="cv-grid">
            <div className="cv-card">
              <h4>Mean Accuracy</h4>
              <div className="cv-value">{(cv_results.mean_accuracy * 100).toFixed(1)}%</div>
              <div className="cv-std">± {(cv_results.std_accuracy * 100).toFixed(1)}%</div>
            </div>
            <div className="cv-chart">
              <h4>Fold Performance</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={cv_results.fold_accuracies.map((acc, i) => ({
                  fold: `Fold ${i + 1}`,
                  accuracy: acc * 100
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="fold" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Accuracy']} />
                  <Bar dataKey="accuracy" fill="#2196F3" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Confusion Matrix */}
      {test_results && test_results.confusion_matrix && (
        <div className="confusion-matrix">
          <h3>🔍 Confusion Matrix</h3>
          <div className="confusion-grid">
            <table className="confusion-table">
              <thead>
                <tr>
                  <th></th>
                  {Object.values(stateNames).map(name => (
                    <th key={name}>{name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {test_results.confusion_matrix.map((row, i) => (
                  <tr key={i}>
                    <td className="state-label">{stateNames[i]}</td>
                    {row.map((cell, j) => (
                      <td 
                        key={j} 
                        className="confusion-cell"
                        style={{ 
                          backgroundColor: `rgba(33, 150, 243, ${cell / Math.max(...row)})`,
                          color: cell > Math.max(...row) / 2 ? 'white' : 'black'
                        }}
                      >
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Cognitive States Distribution */}
      <div className="states-distribution">
        <h3>🧭 Cognitive States Classification</h3>
        <div className="states-grid">
          {Object.entries(stateNames).map(([id, name]) => (
            <div key={id} className="state-card" style={{ borderColor: stateColors[id] }}>
              <div className="state-icon" style={{ backgroundColor: stateColors[id] }}>
                {id}
              </div>
              <h4>{name}</h4>
              <p>EEG Pattern Classification</p>
            </div>
          ))}
        </div>
      </div>

      {/* Training Summary */}
      <div className="training-summary">
        <h3>📋 Training Summary</h3>
        <div className="summary-content">
          <p><strong>Model Type:</strong> CNN-LSTM Hybrid for EEG Classification</p>
          <p><strong>Purpose:</strong> Real-time cognitive state prediction from EEG signals</p>
          <p><strong>Training Method:</strong> Supervised learning with early stopping</p>
          <p><strong>Validation:</strong> Cross-validation with stratified splits</p>
          <p><strong>Performance:</strong> Achieved {((metrics?.test_accuracy || 0) * 100).toFixed(1)}% accuracy on test set</p>
          <p><strong>Last Updated:</strong> {new Date(trainingData.timestamp).toLocaleString()}</p>
        </div>
      </div>
    </div>
  );
};

export default TrainingVisualization;
