#!/usr/bin/env python3
"""
Training script for CNN-LSTM EEG classifier.
Generates synthetic EEG data and trains the model with proper loss minimization.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import CNNLSTMHybrid, EEGClassifier
from utils import generate_mock_eeg_data, create_channel_montage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training parameters
N_CHANNELS = 32
N_TIMEPOINTS = 256
N_CLASSES = 5
SAMPLING_RATE = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# State labels
STATE_LABELS = {
    0: "Relaxed",
    1: "Focused", 
    2: "Stressed",
    3: "High Load",
    4: "Low Load"
}


def generate_synthetic_eeg_data(n_samples=1000, n_channels=32, n_timepoints=256, sampling_rate=128):
    """
    Generate synthetic EEG data with realistic patterns for different cognitive states.
    
    Args:
        n_samples: Number of samples to generate
        n_channels: Number of EEG channels
        n_timepoints: Number of time points per sample
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (eeg_data, labels)
    """
    logger.info(f"Generating {n_samples} synthetic EEG samples...")
    
    eeg_data = []
    labels = []
    
    for i in range(n_samples):
        # Randomly assign a cognitive state
        state = np.random.randint(0, N_CLASSES)
        labels.append(state)
        
        # Generate time axis
        t = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)
        
        # Initialize EEG data for this sample
        sample_data = np.zeros((n_channels, n_timepoints))
        
        # Generate different patterns based on cognitive state
        if state == 0:  # Relaxed
            # Higher alpha power (8-13 Hz), lower beta
            alpha_amplitude = np.random.uniform(10, 20)
            beta_amplitude = np.random.uniform(3, 8)
            theta_amplitude = np.random.uniform(5, 12)
            gamma_amplitude = np.random.uniform(1, 4)
        elif state == 1:  # Focused
            # Balanced alpha and beta, some gamma
            alpha_amplitude = np.random.uniform(8, 15)
            beta_amplitude = np.random.uniform(10, 18)
            theta_amplitude = np.random.uniform(3, 8)
            gamma_amplitude = np.random.uniform(2, 6)
        elif state == 2:  # Stressed
            # Higher beta and gamma, lower alpha
            alpha_amplitude = np.random.uniform(3, 8)
            beta_amplitude = np.random.uniform(15, 25)
            theta_amplitude = np.random.uniform(2, 6)
            gamma_amplitude = np.random.uniform(5, 12)
        elif state == 3:  # High Load
            # Very high beta and gamma, low alpha
            alpha_amplitude = np.random.uniform(2, 6)
            beta_amplitude = np.random.uniform(20, 30)
            theta_amplitude = np.random.uniform(1, 4)
            gamma_amplitude = np.random.uniform(8, 15)
        else:  # Low Load (state 4)
            # Low overall activity, higher alpha
            alpha_amplitude = np.random.uniform(8, 16)
            beta_amplitude = np.random.uniform(2, 6)
            theta_amplitude = np.random.uniform(4, 10)
            gamma_amplitude = np.random.uniform(0.5, 3)
        
        # Generate signals for each channel
        for ch in range(n_channels):
            # Base signal with noise
            signal = np.random.randn(n_timepoints) * 2
            
            # Add frequency components based on cognitive state
            # Alpha waves (8-13 Hz)
            alpha_freq = np.random.uniform(8, 13)
            signal += alpha_amplitude * np.sin(2 * np.pi * alpha_freq * t)
            
            # Beta waves (13-30 Hz)
            beta_freq = np.random.uniform(13, 30)
            signal += beta_amplitude * np.sin(2 * np.pi * beta_freq * t)
            
            # Theta waves (4-8 Hz)
            theta_freq = np.random.uniform(4, 8)
            signal += theta_amplitude * np.sin(2 * np.pi * theta_freq * t)
            
            # Gamma waves (30-100 Hz)
            gamma_freq = np.random.uniform(30, 50)  # Limited to avoid aliasing
            signal += gamma_amplitude * np.sin(2 * np.pi * gamma_freq * t)
            
            # Add some channel-specific variation
            signal *= np.random.uniform(0.8, 1.2)
            
            sample_data[ch] = signal
        
        eeg_data.append(sample_data)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{n_samples} samples")
    
    return np.array(eeg_data), np.array(labels)


def train_model():
    """Train the CNN-LSTM model with synthetic data."""
    
    logger.info("Starting model training...")
    
    # Generate synthetic training data
    X_train, y_train = generate_synthetic_eeg_data(n_samples=2000)
    X_val, y_val = generate_synthetic_eeg_data(n_samples=400)
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = CNNLSTMHybrid(N_CHANNELS, N_TIMEPOINTS, N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log progress
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            logger.info(f"Epoch {epoch+1}/{EPOCHS}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    # Final evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, all_predictions, 
                                    target_names=[STATE_LABELS[i] for i in range(N_CLASSES)]))
    
    # Save trained model
    model_save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_model.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_channels': N_CHANNELS,
            'n_timepoints': N_TIMEPOINTS,
            'n_classes': N_CLASSES
        },
        'training_metrics': {
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    }, model_save_path)
    
    logger.info(f"Model saved to: {model_save_path}")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'training_curves.png')
    plt.savefig(plot_save_path)
    logger.info(f"Training curves saved to: {plot_save_path}")
    
    return model


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train the model
    trained_model = train_model()
    
    logger.info("Training completed successfully!")
