#!/usr/bin/env python3
"""
Quick Training Script for CNN-LSTM EEG Classifier
Simplified version for testing with smaller dataset and fewer epochs.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import json
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import CNNLSTMHybrid
from utils import generate_mock_eeg_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Training parameters (reduced for quick testing)
N_CHANNELS = 32
N_TIMEPOINTS = 256
N_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5

# State labels
STATE_LABELS = {
    0: "Relaxed",
    1: "Focused", 
    2: "Stressed",
    3: "High Load",
    4: "Low Load"
}


def generate_quick_eeg_data(n_samples=500):
    """Generate smaller dataset for quick testing."""
    logger.info(f"Generating {n_samples} EEG samples for quick testing...")
    
    eeg_data = []
    labels = []
    
    # Ensure balanced classes
    samples_per_class = n_samples // N_CLASSES
    
    for class_id in range(N_CLASSES):
        for i in range(samples_per_class):
            # Generate time axis
            t = np.linspace(0, N_TIMEPOINTS / 128, N_TIMEPOINTS)
            
            # Initialize EEG data
            sample_data = np.zeros((N_CHANNELS, N_TIMEPOINTS))
            
            # Generate different patterns based on cognitive state
            if class_id == 0:  # Relaxed
                alpha_amplitude = np.random.uniform(15, 25)
                beta_amplitude = np.random.uniform(3, 8)
            elif class_id == 1:  # Focused
                alpha_amplitude = np.random.uniform(10, 18)
                beta_amplitude = np.random.uniform(12, 20)
            elif class_id == 2:  # Stressed
                alpha_amplitude = np.random.uniform(5, 10)
                beta_amplitude = np.random.uniform(18, 28)
            elif class_id == 3:  # High Load
                alpha_amplitude = np.random.uniform(3, 7)
                beta_amplitude = np.random.uniform(22, 32)
            else:  # Low Load
                alpha_amplitude = np.random.uniform(12, 20)
                beta_amplitude = np.random.uniform(2, 6)
            
            # Generate signals for each channel
            for ch in range(N_CHANNELS):
                signal = np.random.randn(N_TIMEPOINTS) * 2
                
                # Add frequency components
                alpha_freq = np.random.uniform(8, 13)
                signal += alpha_amplitude * np.sin(2 * np.pi * alpha_freq * t)
                
                beta_freq = np.random.uniform(13, 30)
                signal += beta_amplitude * np.sin(2 * np.pi * beta_freq * t)
                
                sample_data[ch] = signal
            
            eeg_data.append(sample_data)
            labels.append(class_id)
    
    # Shuffle the data
    indices = np.random.permutation(len(eeg_data))
    eeg_data = np.array(eeg_data)[indices]
    labels = np.array(labels)[indices]
    
    logger.info(f"Generated data shape: {eeg_data.shape}")
    logger.info(f"Class distribution: {np.bincount(labels)}")
    
    return eeg_data, labels


def train_quick_model():
    """Train model with quick settings."""
    logger.info("Starting quick model training...")
    
    # Generate data
    X, y = generate_quick_eeg_data(n_samples=500)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = CNNLSTMHybrid(N_CHANNELS, N_TIMEPOINTS, N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
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
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log progress
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            logger.info(f"Epoch {epoch+1}/{EPOCHS}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    # Final evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    test_accuracy = accuracy_score(all_labels, all_predictions)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("QUICK TRAINING RESULTS")
    logger.info("="*50)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"Final Train Accuracy: {train_accuracies[-1]:.2f}%")
    logger.info(f"Final Val Accuracy: {val_accuracies[-1]:.2f}%")
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, all_predictions, 
                                    target_names=[STATE_LABELS[i] for i in range(N_CLASSES)]))
    
    # Save model
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'trained_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_channels': N_CHANNELS,
            'n_timepoints': N_TIMEPOINTS,
            'n_classes': N_CLASSES
        },
        'training_metrics': {
            'best_val_loss': best_val_loss,
            'test_accuracy': test_accuracy,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        },
        'training_params': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE
        },
        'timestamp': datetime.now().isoformat()
    }, model_path)
    
    logger.info(f"Model saved to: {model_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
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
    plot_save_path = os.path.join(save_dir, 'quick_training_curves.png')
    plt.savefig(plot_save_path)
    logger.info(f"Training curves saved to: {plot_save_path}")
    
    logger.info("✅ Quick training completed successfully!")
    
    return model, test_accuracy


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train the model
    trained_model, accuracy = train_quick_model()
    
    logger.info(f"\n🎯 Final Test Accuracy: {accuracy:.4f}")
    logger.info("🎉 Model is ready for deployment!")
