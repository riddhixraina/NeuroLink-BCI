"""
Utility functions for NeuroLink-BCI
Common helper functions for data processing, visualization, and model utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import json
import pickle
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_model(model, filepath: str, metadata: Optional[Dict] = None):
    """
    Save a trained model with metadata.
    
    Args:
        model: Trained model object
        filepath: Path to save the model
        metadata: Additional metadata to save
    """
    logger.info(f"Saving model to {filepath}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat()
    }, filepath)
    
    logger.info("Model saved successfully")


def load_model(filepath: str):
    """
    Load a trained model with metadata.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Dictionary containing model state and metadata
    """
    logger.info(f"Loading model from {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    logger.info("Model loaded successfully")
    return checkpoint


def save_features(features: Dict[str, np.ndarray], filepath: str):
    """
    Save extracted features to file.
    
    Args:
        features: Dictionary containing feature arrays
        filepath: Path to save features
    """
    logger.info(f"Saving features to {filepath}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save features
    np.savez(filepath, **features)
    
    logger.info("Features saved successfully")


def load_features(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load features from file.
    
    Args:
        filepath: Path to the saved features
        
    Returns:
        Dictionary containing feature arrays
    """
    logger.info(f"Loading features from {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Features file not found: {filepath}")
    
    features = np.load(filepath)
    features_dict = {key: features[key] for key in features.keys()}
    
    logger.info("Features loaded successfully")
    return features_dict


def plot_eeg_signals(eeg_data: np.ndarray, 
                    channel_names: List[str],
                    sampling_rate: int = 128,
                    start_time: float = 0,
                    duration: float = 5,
                    title: str = "EEG Signals"):
    """
    Plot EEG signals for visualization.
    
    Args:
        eeg_data: EEG data (channels, time_points)
        channel_names: List of channel names
        sampling_rate: Sampling rate in Hz
        start_time: Start time in seconds
        duration: Duration to plot in seconds
        title: Plot title
    """
    n_channels, n_timepoints = eeg_data.shape
    
    # Calculate time axis
    time_points = int(duration * sampling_rate)
    start_sample = int(start_time * sampling_rate)
    end_sample = min(start_sample + time_points, n_timepoints)
    
    time_axis = np.linspace(start_time, start_time + duration, end_sample - start_sample)
    
    # Create subplots
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels))
    if n_channels == 1:
        axes = [axes]
    
    # Plot each channel
    for i, (channel_data, channel_name) in enumerate(zip(eeg_data, channel_names)):
        axes[i].plot(time_axis, channel_data[start_sample:end_sample])
        axes[i].set_ylabel(f"{channel_name}\n(μV)")
        axes[i].grid(True, alpha=0.3)
        
        if i == n_channels - 1:
            axes[i].set_xlabel("Time (s)")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_frequency_spectrum(eeg_data: np.ndarray,
                          channel_names: List[str],
                          sampling_rate: int = 128,
                          channel_idx: int = 0):
    """
    Plot frequency spectrum of EEG data.
    
    Args:
        eeg_data: EEG data (channels, time_points)
        channel_names: List of channel names
        sampling_rate: Sampling rate in Hz
        channel_idx: Channel index to plot
    """
    from scipy.signal import welch
    
    # Calculate PSD
    freqs, psd = welch(eeg_data[channel_idx, :], fs=sampling_rate, nperseg=256)
    
    # Plot spectrum
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'Frequency Spectrum - {channel_names[channel_idx]}')
    plt.grid(True, alpha=0.3)
    
    # Mark frequency bands
    frequency_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }
    
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for (band, (low, high)), color in zip(frequency_bands.items(), colors):
        plt.axvspan(low, high, alpha=0.2, color=color, label=band)
    
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         class_names: List[str],
                         title: str = "Confusion Matrix"):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_training_history(history: Dict):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_history']['loss'], label='Training Loss')
    ax1.plot(history['val_history']['loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_history']['accuracy'], label='Training Accuracy')
    ax2.plot(history['val_history']['accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_experiment_log(experiment_name: str, 
                         config: Dict,
                         results: Dict,
                         filepath: str = "experiments"):
    """
    Create an experiment log file.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        results: Experiment results
        filepath: Directory to save the log
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.json"
    log_path = os.path.join(filepath, log_filename)
    
    # Create directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)
    
    # Create log entry
    log_entry = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'config': config,
        'results': results
    }
    
    # Save log
    with open(log_path, 'w') as f:
        json.dump(log_entry, f, indent=2, default=str)
    
    logger.info(f"Experiment log saved to {log_path}")


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Class labels
        
    Returns:
        Dictionary mapping class indices to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def normalize_data(data: np.ndarray, method: str = 'z_score') -> np.ndarray:
    """
    Normalize data using various methods.
    
    Args:
        data: Input data
        method: Normalization method ('z_score', 'minmax', 'robust')
        
    Returns:
        Normalized data
    """
    if method == 'z_score':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return (data - median) / mad
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def generate_mock_eeg_data(n_samples: int = 100,
                          n_channels: int = 32,
                          n_timepoints: int = 256,
                          sampling_rate: int = 128,
                          noise_level: float = 0.1) -> np.ndarray:
    """
    Generate mock EEG data for testing.
    
    Args:
        n_samples: Number of samples
        n_channels: Number of channels
        n_timepoints: Number of time points
        sampling_rate: Sampling rate in Hz
        noise_level: Noise level
        
    Returns:
        Mock EEG data
    """
    # Generate time axis
    t = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)
    
    # Initialize data array
    eeg_data = np.zeros((n_samples, n_channels, n_timepoints))
    
    for i in range(n_samples):
        for j in range(n_channels):
            # Generate synthetic EEG with multiple frequency components
            signal = (
                10 * np.sin(2 * np.pi * 10 * t) +  # Alpha waves
                5 * np.sin(2 * np.pi * 20 * t) +   # Beta waves
                3 * np.sin(2 * np.pi * 5 * t) +    # Theta waves
                noise_level * np.random.randn(n_timepoints)  # Noise
            )
            
            eeg_data[i, j, :] = signal
    
    return eeg_data


def create_channel_montage(n_channels: int) -> List[str]:
    """
    Create channel names for EEG montage.
    
    Args:
        n_channels: Number of channels
        
    Returns:
        List of channel names
    """
    if n_channels == 32:
        return [
            'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'CP5',
            'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2',
            'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'CP6',
            'CP2', 'P4', 'P8', 'PO4', 'O2', 'Fc5', 'Fc1', 'Cz'
        ]
    elif n_channels == 62:
        return [
            'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
            'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
            'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
            'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
        ]
    elif n_channels == 64:
        return [f'Ch{i:02d}' for i in range(1, 65)]
    else:
        return [f'Ch{i:02d}' for i in range(1, n_channels + 1)]


def main():
    """Example usage of utility functions."""
    # Generate mock data
    eeg_data = generate_mock_eeg_data(n_samples=10, n_channels=32, n_timepoints=256)
    channel_names = create_channel_montage(32)
    
    print("Generated mock EEG data:")
    print(f"Shape: {eeg_data.shape}")
    print(f"Channel names: {channel_names[:5]}...")
    
    # Test normalization
    normalized_data = normalize_data(eeg_data[0, 0, :], method='z_score')
    print(f"Normalized data stats: mean={np.mean(normalized_data):.3f}, std={np.std(normalized_data):.3f}")
    
    # Test class weights
    y = np.random.randint(0, 3, 100)
    weights = calculate_class_weights(y)
    print(f"Class weights: {weights}")


if __name__ == "__main__":
    main()
