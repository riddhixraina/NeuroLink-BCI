"""
Ultra-minimal Flask Backend for Railway Deployment
This version prioritizes startup speed and reliability with Socket.IO support
"""

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import threading
import time
import numpy as np
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
CORS(app, origins=[
    "https://neuro-link-bci.vercel.app",
    "https://neuro-link-bci-git-main-riddhixrainas-projects.vercel.app",
    "https://neuro-link-exetr1b97-riddhixrainas-projects.vercel.app",
    "http://localhost:3000",
    "http://localhost:3001"
])
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    logger=True, 
    engineio_logger=True,
    always_connect=True,
    ping_timeout=60,
    ping_interval=25
)

# Global variables
streaming_active = False
streaming_thread = None

def generate_mock_eeg_data():
    """Generate realistic mock EEG data"""
    n_channels = 32
    n_samples = 256
    sampling_rate = 256
    
    # Generate realistic EEG patterns
    time_points = np.linspace(0, 1, n_samples)
    
    # Different frequency bands
    alpha = np.sin(2 * np.pi * 10 * time_points)  # 10 Hz alpha
    beta = np.sin(2 * np.pi * 20 * time_points)    # 20 Hz beta
    theta = np.sin(2 * np.pi * 6 * time_points)   # 6 Hz theta
    gamma = np.sin(2 * np.pi * 40 * time_points)  # 40 Hz gamma
    
    # Generate data for each channel
    eeg_data = []
    channel_names = [f'Channel_{i+1}' for i in range(n_channels)]
    
    for i in range(n_channels):
        # Mix different frequency bands with noise
        noise = np.random.normal(0, 0.1, n_samples)
        signal = alpha + 0.5*beta + 0.3*theta + 0.2*gamma + noise
        
        # Add channel-specific variations
        signal += np.sin(2 * np.pi * (5 + i*0.5) * time_points) * 0.1
        
        eeg_data.append(signal.tolist())
    
    return {
        'eeg_data': eeg_data,
        'channel_names': channel_names,
        'sampling_rate': sampling_rate,
        'timestamp': datetime.now().isoformat()
    }

def generate_mock_prediction():
    """Generate mock cognitive state prediction"""
    states = ['Relaxed', 'Focused', 'Stressed', 'High Load', 'Low Load']
    predicted_state = np.random.choice(states)
    confidence = np.random.uniform(0.6, 0.95)
    
    return {
        'predicted_state': predicted_state,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    }

def streaming_worker():
    """Background worker for streaming mock EEG data"""
    global streaming_active
    
    while streaming_active:
        try:
            # Generate mock data
            eeg_data = generate_mock_eeg_data()
            prediction = generate_mock_prediction()
            
            # Combine data
            combined_data = {
                'eeg_data': eeg_data['eeg_data'],
                'channel_names': eeg_data['channel_names'],
                'sampling_rate': eeg_data['sampling_rate'],
                'prediction': prediction,
                'timestamp': eeg_data['timestamp']
            }
            
            # Emit to connected clients
            socketio.emit('eeg_data', combined_data)
            
            # Wait before next update
            time.sleep(0.1)  # 10 Hz update rate
            
        except Exception as e:
            print(f"Error in streaming worker: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return jsonify({
        'message': 'NeuroLink-BCI Backend API',
        'status': 'running',
        'version': '1.0.0',
        'mode': 'ultra-minimal'
    })

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'streaming': streaming_active,
        'version': '1.0.0',
        'mode': 'ultra-minimal',
        'cors_origins': [
            "https://neuro-link-bci.vercel.app",
            "https://neuro-link-bci-git-main-riddhixrainas-projects.vercel.app",
            "https://neuro-link-exetr1b97-riddhixrainas-projects.vercel.app"
        ]
    })

@app.route('/api/test')
def test():
    return jsonify({
        'message': 'Backend connection test successful',
        'timestamp': datetime.now().isoformat(),
        'origin': request.headers.get('Origin', 'No origin header'),
        'user_agent': request.headers.get('User-Agent', 'No user agent')
    })

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'running',
        'streaming': streaming_active,
        'connected_clients': 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/start_streaming', methods=['POST'])
def start_streaming():
    global streaming_active, streaming_thread
    
    if not streaming_active:
        streaming_active = True
        streaming_thread = threading.Thread(target=streaming_worker)
        streaming_thread.daemon = True
        streaming_thread.start()
        
        return jsonify({'status': 'started', 'message': 'EEG streaming started'})
    else:
        return jsonify({'status': 'already_running', 'message': 'Streaming already active'})

@app.route('/api/stop_streaming', methods=['POST'])
def stop_streaming():
    global streaming_active
    
    streaming_active = False
    return jsonify({'status': 'stopped', 'message': 'EEG streaming stopped'})

@app.route('/api/set_cognitive_state', methods=['POST'])
def set_cognitive_state():
    data = request.get_json()
    state = data.get('state', 0)
    return jsonify({'status': 'success', 'state': state})

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    return jsonify({
        'model_exists': True,
        'model_config': {
            'n_channels': 32,
            'n_timepoints': 256,
            'n_classes': 5
        },
        'training_params': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        'metrics': {
            'test_accuracy': 0.77,
            'best_val_loss': 0.45,
            'final_train_accuracy': 0.82,
            'final_val_accuracy': 0.75
        },
        'training_curves': {
            'epochs': list(range(1, 51)),
            'train_losses': [0.8 - i*0.01 for i in range(50)],
            'val_losses': [0.85 - i*0.008 for i in range(50)],
            'train_accuracies': [0.3 + i*0.01 for i in range(50)],
            'val_accuracies': [0.25 + i*0.009 for i in range(50)]
        },
        'test_results': {
            'accuracy': 0.77,
            'precision': 0.75,
            'recall': 0.78,
            'f1_score': 0.76,
            'confusion_matrix': [[15, 2, 1, 0, 1], [1, 18, 2, 0, 0], [0, 1, 16, 2, 0], [0, 0, 1, 17, 1], [1, 0, 0, 1, 16]]
        },
        'cv_results': {
            'mean_accuracy': 0.76,
            'std_accuracy': 0.03,
            'fold_accuracies': [0.75, 0.77, 0.76, 0.78, 0.74]
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/training/progress', methods=['GET'])
def get_training_progress():
    return jsonify({
        'is_training': False,
        'progress': 100,
        'message': 'Training completed'
    })

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    print(f'Client transport: {request.transport}')
    print(f'Client namespace: {request.namespace}')
    emit('status', {'status': 'connected', 'server_version': '1.0.0'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

@socketio.on('start_streaming')
def handle_start_streaming():
    print('Client requested streaming start')
    # Streaming is handled by the API endpoint

@socketio.on('stop_streaming')
def handle_stop_streaming():
    print('Client requested streaming stop')
    # Streaming stop is handled by the API endpoint

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Ultra-Minimal NeuroLink-BCI Backend on port {port}")
    print(f"Environment: PORT={port}, FLASK_ENV={os.environ.get('FLASK_ENV', 'development')}")
    print("Initializing SocketIO server...")
    
    try:
        # Start the app with SocketIO
        print("Starting SocketIO server...")
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"Error starting SocketIO server: {e}")
        print("Falling back to simple Flask app...")
        # Fallback to simple Flask app if SocketIO fails
        app.run(host='0.0.0.0', port=port, debug=False)
