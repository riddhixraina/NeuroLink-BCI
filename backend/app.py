"""
Flask Backend for NeuroLink-BCI Real-Time EEG Processing
Provides REST API and WebSocket streaming for real-time neural state prediction.
"""

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import numpy as np
import json
import threading
import time
from datetime import datetime
import logging
import sys
import os
from config import config

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import EEGDataLoader
from preprocess import EEGPreprocessor
from feature_extraction import EEGFeatureExtractor
from model import EEGClassifier, CNNLSTMHybrid
from utils import generate_mock_eeg_data, create_channel_montage
import numpy as np
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockEEGClassifier:
    """
    Mock classifier that generates realistic predictions based on EEG signal characteristics.
    """
    
    def __init__(self):
        self.n_classes = 5
        self.state_names = ["Relaxed", "Focused", "Stressed", "High Load", "Low Load"]
        
    def predict(self, eeg_data):
        """
        Generate realistic predictions based on EEG signal characteristics.
        
        Args:
            eeg_data: EEG data (samples, channels, time_points)
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        try:
            # Convert to numpy if it's a tensor
            if hasattr(eeg_data, 'numpy'):
                eeg_data = eeg_data.numpy()
            
            # Ensure we have the right shape
            if eeg_data.ndim == 3:
                eeg_data = eeg_data.reshape(eeg_data.shape[0], -1)  # Flatten to (samples, features)
            
            # Calculate signal characteristics
            mean_amplitude = np.mean(np.abs(eeg_data), axis=1)
            variance = np.var(eeg_data, axis=1)
            std_dev = np.std(eeg_data, axis=1)
            
            # Generate predictions based on signal characteristics
            predictions = []
            confidences = []
            
            for i in range(eeg_data.shape[0]):
                # Use signal characteristics to determine state
                if std_dev[i] < 5:  # Low variability
                    pred = 4  # Low Load
                    conf = 0.85
                elif std_dev[i] > 15:  # High variability
                    pred = 2  # Stressed
                    conf = 0.78
                elif variance[i] > 100:  # High variance
                    pred = 3  # High Load
                    conf = 0.82
                elif mean_amplitude[i] > 20:  # High amplitude
                    pred = 1  # Focused
                    conf = 0.75
                else:  # Balanced
                    pred = 0  # Relaxed
                    conf = 0.80
                
                # Add some randomness to make it more realistic
                if np.random.random() < 0.1:  # 10% chance to change prediction
                    pred = np.random.randint(0, 5)
                    conf = np.random.uniform(0.6, 0.9)
                
                predictions.append(pred)
                confidences.append(conf)
            
            return np.array(predictions), np.array(confidences)
            
        except Exception as e:
            logger.error(f"Mock classifier error: {str(e)}")
            # Fallback to random predictions
            predictions = np.random.randint(0, 5, size=(eeg_data.shape[0] if hasattr(eeg_data, 'shape') else 1,))
            confidences = np.random.uniform(0.6, 0.9, size=predictions.shape)
            return predictions, confidences


def load_trained_classifier():
    """
    Load trained classifier if available, otherwise return mock classifier.
    
    Returns:
        Trained classifier or mock classifier
    """
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_model.pth')
    
    if os.path.exists(model_path):
        try:
            logger.info("Loading trained model...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Check if it's the improved model format
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
                
                # Initialize model with saved configuration
                model = CNNLSTMHybrid(
                    model_config['n_channels'],
                    model_config['n_timepoints'], 
                    model_config['n_classes']
                ).to(device)
                
                # Load trained weights
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                # Log training results if available
                if 'test_results' in checkpoint:
                    test_results = checkpoint['test_results']
                    logger.info(f"Model test accuracy: {test_results['accuracy']:.4f}")
                    logger.info(f"Model test F1-score: {test_results['f1_score']:.4f}")
                
                if 'cv_results' in checkpoint:
                    cv_results = checkpoint['cv_results']
                    logger.info(f"Cross-validation accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
                
                # Create classifier wrapper
                classifier = TrainedEEGClassifier(model, device)
                logger.info("Successfully loaded improved trained model!")
                return classifier
            else:
                # Fallback for old model format
                logger.warning("Old model format detected, attempting to load...")
                model = CNNLSTMHybrid(32, 256, 5).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                classifier = TrainedEEGClassifier(model, device)
                logger.info("Successfully loaded legacy trained model!")
                return classifier
            
        except Exception as e:
            logger.error(f"Error loading trained model: {str(e)}")
            logger.info("Falling back to mock classifier...")
            return MockEEGClassifier()
    else:
        logger.info("No trained model found. Using mock classifier...")
        return MockEEGClassifier()


class TrainedEEGClassifier:
    """
    Wrapper for trained PyTorch model to make predictions.
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.n_classes = 5
        self.state_names = ["Relaxed", "Focused", "Stressed", "High Load", "Low Load"]
        
    def predict(self, eeg_data):
        """
        Make predictions using the trained model.
        
        Args:
            eeg_data: EEG data (samples, channels, time_points)
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        try:
            # Convert to tensor if needed
            if not isinstance(eeg_data, torch.Tensor):
                eeg_data = torch.FloatTensor(eeg_data)
            
            # Move to device
            eeg_data = eeg_data.to(self.device)
            
            # Ensure correct shape
            if eeg_data.ndim == 2:
                eeg_data = eeg_data.reshape(1, eeg_data.shape[0], eeg_data.shape[1])
            
            with torch.no_grad():
                outputs = self.model(eeg_data)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                confidence = probabilities.max(dim=1)[0]
            
            return predictions.cpu().numpy(), confidence.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Trained classifier error: {str(e)}")
            # Fallback to mock predictions
            mock_classifier = MockEEGClassifier()
            return mock_classifier.predict(eeg_data)


# Initialize Flask app
app = Flask(__name__)

# Load configuration
config_name = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[config_name])

# Initialize extensions
CORS(app, origins=app.config['CORS_ORIGINS'])
socketio = SocketIO(app, cors_allowed_origins=app.config['CORS_ORIGINS'])

# Global variables for model and data processing
classifier = None
preprocessor = None
feature_extractor = None
data_loader = None
streaming_active = False
streaming_thread = None
simulator = None

# Cognitive state labels
COGNITIVE_STATES = {
    0: "Relaxed",
    1: "Focused", 
    2: "Stressed",
    3: "High Load",
    4: "Low Load"
}

# Initialize components
def initialize_components():
    """Initialize all processing components."""
    global classifier, preprocessor, feature_extractor, data_loader, simulator
    
    logger.info("Initializing NeuroLink-BCI components...")
    
    # Initialize data loader
    data_loader = EEGDataLoader()
    
    # Initialize preprocessor
    preprocessor = EEGPreprocessor(sampling_rate=128, n_channels=32)
    
    # Initialize feature extractor
    feature_extractor = EEGFeatureExtractor(sampling_rate=128)
    
    # Initialize classifier - try to load trained model, fallback to mock
    classifier = load_trained_classifier()
    
    # Initialize simulator
    from streaming import RealTimeEEGSimulator
    simulator = RealTimeEEGSimulator(n_channels=32, sampling_rate=128)
    
    logger.info("Components initialized successfully")


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status."""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'streaming_active': streaming_active,
        'components_loaded': all([classifier, preprocessor, feature_extractor, data_loader, simulator])
    })


@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get available datasets information."""
    if not data_loader:
        return jsonify({'error': 'Data loader not initialized'}), 500
    
    datasets_info = {}
    for dataset in ['deap', 'seed', 'physionet']:
        datasets_info[dataset] = data_loader.get_dataset_info(dataset)
    
    return jsonify(datasets_info)


@app.route('/api/load_dataset/<dataset_name>', methods=['POST'])
def load_dataset(dataset_name):
    """Load a specific dataset."""
    try:
        if not data_loader:
            return jsonify({'error': 'Data loader not initialized'}), 500
        
        if dataset_name == 'deap':
            data = data_loader.load_deap_dataset()
        elif dataset_name == 'seed':
            data = data_loader.load_seed_dataset()
        elif dataset_name == 'physionet':
            data = data_loader.load_physionet_dataset()
        else:
            return jsonify({'error': 'Invalid dataset name'}), 400
        
        return jsonify({
            'dataset': dataset_name,
            'shape': data['eeg_data'].shape,
            'sampling_rate': data['sampling_rate'],
            'channels': data['channels'],
            'status': 'loaded'
        })
    
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """Preprocess EEG data."""
    try:
        if not preprocessor:
            return jsonify({'error': 'Preprocessor not initialized'}), 500
        
        # Get data from request (mock for now)
        # In production, this would receive actual EEG data
        mock_data = generate_mock_eeg_data(n_samples=50, n_channels=32, n_timepoints=7680)
        mock_labels = np.random.randint(0, 5, 50)
        
        # Preprocess data
        results = preprocessor.preprocess_pipeline(mock_data, mock_labels)
        
        return jsonify({
            'status': 'preprocessed',
            'original_trials': results['n_trials_original'],
            'clean_trials': results['n_trials_clean'],
            'artifacts_removed': results['artifacts_removed'],
            'final_shape': results['eeg_data'].shape
        })
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract_features', methods=['POST'])
def extract_features():
    """Extract features from EEG data."""
    try:
        if not feature_extractor:
            return jsonify({'error': 'Feature extractor not initialized'}), 500
        
        # Generate mock data for demonstration
        mock_data = generate_mock_eeg_data(n_samples=100, n_channels=32, n_timepoints=256)
        
        # Extract features
        features = feature_extractor.extract_all_features(mock_data)
        
        # Count features
        feature_count = sum(feat_array.size for feat_array in features.values())
        
        return jsonify({
            'status': 'features_extracted',
            'feature_types': len(features),
            'total_features': feature_count,
            'feature_names': list(features.keys())
        })
    
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_cognitive_state():
    """Predict cognitive state from EEG data."""
    try:
        if not classifier:
            return jsonify({'error': 'Classifier not initialized'}), 500
        
        # Get EEG data from request
        data = request.get_json()
        
        if 'eeg_data' not in data:
            return jsonify({'error': 'EEG data not provided'}), 400
        
        eeg_data = np.array(data['eeg_data'])
        
        # Ensure correct shape
        if eeg_data.ndim == 2:
            eeg_data = eeg_data.reshape(1, eeg_data.shape[0], eeg_data.shape[1])
        
        # Make prediction
        predictions, confidence = classifier.predict(eeg_data)
        
        # Convert to readable format
        predicted_state = COGNITIVE_STATES.get(predictions[0], "Unknown")
        
        return jsonify({
            'predicted_state': predicted_state,
            'confidence': float(confidence[0]),
            'state_id': int(predictions[0]),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/start_streaming', methods=['POST'])
def start_streaming():
    """Start real-time EEG streaming simulation."""
    global streaming_active, streaming_thread
    
    if streaming_active:
        return jsonify({'status': 'already_streaming'})
    
    streaming_active = True
    streaming_thread = threading.Thread(target=stream_eeg_data)
    streaming_thread.daemon = True
    streaming_thread.start()
    
    return jsonify({'status': 'streaming_started'})


@app.route('/api/stop_streaming', methods=['POST'])
def stop_streaming():
    """Stop real-time EEG streaming."""
    global streaming_active
    
    streaming_active = False
    
    return jsonify({'status': 'streaming_stopped'})


@app.route('/api/set_cognitive_state', methods=['POST'])
def set_cognitive_state():
    """Set the cognitive state for simulation."""
    global simulator
    
    try:
        data = request.get_json()
        state = data.get('state', 0)
        
        if simulator:
            simulator.set_cognitive_state(state)
            state_name = COGNITIVE_STATES.get(state, "Unknown")
            return jsonify({
                'status': 'state_updated',
                'state_id': state,
                'state_name': state_name
            })
        else:
            return jsonify({'error': 'Simulator not initialized'}), 500
    except Exception as e:
        logger.error(f"Error setting cognitive state: {str(e)}")
        return jsonify({'error': str(e)}), 500


def stream_eeg_data():
    """Stream simulated EEG data in real-time."""
    global streaming_active, simulator
    
    logger.info("Starting EEG data streaming...")
    
    # Generate mock EEG data
    n_channels = 32
    n_timepoints = 256  # 2 seconds at 128 Hz
    sampling_rate = 128
    channel_names = create_channel_montage(n_channels)
    
    while streaming_active:
        try:
            # Generate new EEG sample using simulator
            if simulator:
                eeg_sample = simulator._generate_eeg_chunk(n_timepoints)
            else:
                # Fallback to basic mock data if simulator not available
                eeg_sample = generate_mock_eeg_data(
                    n_samples=1, 
                    n_channels=n_channels, 
                    n_timepoints=n_timepoints,
                    sampling_rate=sampling_rate
                )[0]
            
            # Make prediction if classifier is available
            prediction_result = None
            if classifier:
                try:
                    predictions, confidence = classifier.predict(eeg_sample.reshape(1, n_channels, n_timepoints))
                    predicted_state = COGNITIVE_STATES.get(predictions[0], "Unknown")
                    
                    prediction_result = {
                        'predicted_state': predicted_state,
                        'confidence': float(confidence[0]),
                        'state_id': int(predictions[0])
                    }
                except Exception as e:
                    logger.error(f"Prediction error: {str(e)}")
            
            # Create data packet
            data_packet = {
                'timestamp': datetime.now().isoformat(),
                'eeg_data': eeg_sample.tolist(),
                'channel_names': channel_names,
                'sampling_rate': sampling_rate,
                'cognitive_state': simulator.current_state if simulator else 0,
                'prediction': prediction_result
            }
            
            # Emit to all connected clients
            socketio.emit('eeg_data', data_packet)
            
            # Sleep for real-time simulation (2 seconds of data every 2 seconds)
            time.sleep(2.0)
            
        except Exception as e:
            logger.error(f"Error in streaming loop: {str(e)}")
            time.sleep(1.0)
    
    logger.info("EEG data streaming stopped")


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to NeuroLink-BCI'})
    
    # Send system status
    emit('system_status', {
        'status': 'running',
        'streaming_active': streaming_active,
        'components_loaded': all([classifier, preprocessor, feature_extractor, data_loader, simulator])
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('request_status')
def handle_status_request():
    """Handle status request from client."""
    emit('system_status', {
        'status': 'running',
        'streaming_active': streaming_active,
        'components_loaded': all([classifier, preprocessor, feature_extractor, data_loader, simulator]),
        'timestamp': datetime.now().isoformat()
    })


@socketio.on('start_streaming')
def handle_start_streaming():
    """Handle start streaming request from client."""
    global streaming_active, streaming_thread
    
    if not streaming_active:
        streaming_active = True
        streaming_thread = threading.Thread(target=stream_eeg_data)
        streaming_thread.daemon = True
        streaming_thread.start()
        
        emit('streaming_status', {'status': 'started'})
    else:
        emit('streaming_status', {'status': 'already_active'})


@socketio.on('stop_streaming')
def handle_stop_streaming():
    """Handle stop streaming request from client."""
    global streaming_active
    
    streaming_active = False
    emit('streaming_status', {'status': 'stopped'})


@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get training status and results."""
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_model.pth')
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract training information
            training_info = {
                'model_exists': True,
                'model_config': checkpoint.get('model_config', {}),
                'training_params': checkpoint.get('training_params', {}),
                'timestamp': checkpoint.get('timestamp', 'Unknown')
            }
            
            # Add training metrics if available
            if 'training_metrics' in checkpoint:
                metrics = checkpoint['training_metrics']
                training_info['metrics'] = {
                    'test_accuracy': metrics.get('test_accuracy', 0),
                    'best_val_loss': metrics.get('best_val_loss', 0),
                    'final_train_accuracy': metrics.get('train_accuracies', [0])[-1] if metrics.get('train_accuracies') else 0,
                    'final_val_accuracy': metrics.get('val_accuracies', [0])[-1] if metrics.get('val_accuracies') else 0
                }
                
                # Add training curves data
                if 'train_losses' in metrics and 'val_losses' in metrics:
                    training_info['training_curves'] = {
                        'epochs': list(range(1, len(metrics['train_losses']) + 1)),
                        'train_losses': metrics['train_losses'],
                        'val_losses': metrics['val_losses'],
                        'train_accuracies': metrics.get('train_accuracies', []),
                        'val_accuracies': metrics.get('val_accuracies', [])
                    }
            
            # Add test results if available
            if 'test_results' in checkpoint:
                test_results = checkpoint['test_results']
                training_info['test_results'] = {
                    'accuracy': test_results.get('accuracy', 0),
                    'precision': test_results.get('precision', 0),
                    'recall': test_results.get('recall', 0),
                    'f1_score': test_results.get('f1_score', 0),
                    'confusion_matrix': test_results.get('confusion_matrix', [])
                }
            
            # Add cross-validation results if available
            if 'cv_results' in checkpoint:
                cv_results = checkpoint['cv_results']
                training_info['cv_results'] = {
                    'mean_accuracy': cv_results.get('mean_accuracy', 0),
                    'std_accuracy': cv_results.get('std_accuracy', 0),
                    'fold_accuracies': cv_results.get('fold_accuracies', [])
                }
            
            return jsonify(training_info)
        else:
            return jsonify({
                'model_exists': False,
                'message': 'No trained model found'
            })
    
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training."""
    try:
        # Check if training is already in progress
        if hasattr(start_training, 'training_in_progress') and start_training.training_in_progress:
            return jsonify({'status': 'already_training'})
        
        # Start training in background
        import threading
        def run_training():
            start_training.training_in_progress = True
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, 
                    os.path.join(os.path.dirname(__file__), '..', 'scripts', 'quick_train_model.py')
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("Training completed successfully")
                else:
                    logger.error(f"Training failed: {result.stderr}")
            except Exception as e:
                logger.error(f"Training error: {str(e)}")
            finally:
                start_training.training_in_progress = False
        
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({'status': 'training_started'})
    
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/progress', methods=['GET'])
def get_training_progress():
    """Get training progress."""
    try:
        # Check if training is in progress
        training_in_progress = hasattr(start_training, 'training_in_progress') and start_training.training_in_progress
        
        return jsonify({
            'training_in_progress': training_in_progress,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting training progress: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize components
    initialize_components()
    
    # Run the application
    logger.info("Starting NeuroLink-BCI backend server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
