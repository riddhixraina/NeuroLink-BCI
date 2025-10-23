#!/usr/bin/env python3
"""
Improved Training Script for CNN-LSTM EEG Classifier
Implements proper train/test splits, cross-validation, early stopping, and comprehensive evaluation.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import CNNLSTMHybrid, EEGClassifier
from utils import generate_mock_eeg_data, create_channel_montage

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Training parameters
N_CHANNELS = 32
N_TIMEPOINTS = 256
N_CLASSES = 5
SAMPLING_RATE = 128
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
K_FOLDS = 5

# State labels
STATE_LABELS = {
    0: "Relaxed",
    1: "Focused", 
    2: "Stressed",
    3: "High Load",
    4: "Low Load"
}


def generate_realistic_eeg_data(n_samples=3000, n_channels=32, n_timepoints=256, sampling_rate=128):
    """
    Generate more realistic synthetic EEG data with proper class balance and realistic patterns.
    
    Args:
        n_samples: Number of samples to generate
        n_channels: Number of EEG channels
        n_timepoints: Number of time points per sample
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (eeg_data, labels)
    """
    logger.info(f"Generating {n_samples} realistic synthetic EEG samples...")
    
    eeg_data = []
    labels = []
    
    # Ensure balanced classes
    samples_per_class = n_samples // N_CLASSES
    remaining_samples = n_samples % N_CLASSES
    
    for class_id in range(N_CLASSES):
        class_samples = samples_per_class + (1 if class_id < remaining_samples else 0)
        
        for i in range(class_samples):
            # Generate time axis
            t = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)
            
            # Initialize EEG data for this sample
            sample_data = np.zeros((n_channels, n_timepoints))
            
            # Generate different patterns based on cognitive state
            if class_id == 0:  # Relaxed
                # Higher alpha power (8-13 Hz), lower beta, more theta
                alpha_amplitude = np.random.uniform(15, 25)
                beta_amplitude = np.random.uniform(3, 8)
                theta_amplitude = np.random.uniform(8, 15)
                gamma_amplitude = np.random.uniform(1, 3)
                noise_level = 2
            elif class_id == 1:  # Focused
                # Balanced alpha and beta, moderate gamma
                alpha_amplitude = np.random.uniform(10, 18)
                beta_amplitude = np.random.uniform(12, 20)
                theta_amplitude = np.random.uniform(4, 8)
                gamma_amplitude = np.random.uniform(3, 7)
                noise_level = 3
            elif class_id == 2:  # Stressed
                # Higher beta and gamma, lower alpha
                alpha_amplitude = np.random.uniform(5, 10)
                beta_amplitude = np.random.uniform(18, 28)
                theta_amplitude = np.random.uniform(2, 6)
                gamma_amplitude = np.random.uniform(6, 12)
                noise_level = 4
            elif class_id == 3:  # High Load
                # Very high beta and gamma, low alpha
                alpha_amplitude = np.random.uniform(3, 7)
                beta_amplitude = np.random.uniform(22, 32)
                theta_amplitude = np.random.uniform(1, 4)
                gamma_amplitude = np.random.uniform(8, 15)
                noise_level = 5
            else:  # Low Load (state 4)
                # Low overall activity, higher alpha
                alpha_amplitude = np.random.uniform(12, 20)
                beta_amplitude = np.random.uniform(2, 6)
                theta_amplitude = np.random.uniform(6, 12)
                gamma_amplitude = np.random.uniform(0.5, 2)
                noise_level = 1.5
            
            # Generate signals for each channel
            for ch in range(n_channels):
                # Base signal with noise
                signal = np.random.randn(n_timepoints) * noise_level
                
                # Add frequency components based on cognitive state
                # Alpha waves (8-13 Hz)
                alpha_freq = np.random.uniform(8, 13)
                signal += alpha_amplitude * np.sin(2 * np.pi * alpha_freq * t + np.random.uniform(0, 2*np.pi))
                
                # Beta waves (13-30 Hz)
                beta_freq = np.random.uniform(13, 30)
                signal += beta_amplitude * np.sin(2 * np.pi * beta_freq * t + np.random.uniform(0, 2*np.pi))
                
                # Theta waves (4-8 Hz)
                theta_freq = np.random.uniform(4, 8)
                signal += theta_amplitude * np.sin(2 * np.pi * theta_freq * t + np.random.uniform(0, 2*np.pi))
                
                # Gamma waves (30-50 Hz)
                gamma_freq = np.random.uniform(30, 50)
                signal += gamma_amplitude * np.sin(2 * np.pi * gamma_freq * t + np.random.uniform(0, 2*np.pi))
                
                # Add some channel-specific variation and spatial correlation
                channel_factor = np.random.uniform(0.7, 1.3)
                signal *= channel_factor
                
                # Add some artifacts occasionally
                if np.random.random() < 0.1:  # 10% chance of artifacts
                    artifact_start = np.random.randint(0, n_timepoints - 50)
                    artifact_duration = np.random.randint(10, 50)
                    signal[artifact_start:artifact_start + artifact_duration] += np.random.uniform(-20, 20)
                
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


def create_data_loaders(X, y, batch_size=32, test_size=0.2, val_size=0.2, random_state=42):
    """
    Create train, validation, and test data loaders with proper splits.
    
    Args:
        X: EEG data
        y: Labels
        batch_size: Batch size for data loaders
        test_size: Fraction of data for testing
        val_size: Fraction of remaining data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, test_indices)
    """
    logger.info("Creating train/validation/test splits...")
    
    # First split: separate test set
    if test_size is not None:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        # No test split needed (for cross-validation)
        X_temp, y_temp = X, y
        X_test, y_test = None, None
    
    # Second split: separate train and validation from remaining data
    if test_size is not None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=random_state, stratify=y_temp
        )
    else:
        # Direct validation split when no test set
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    if X_test is not None:
        logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Handle test set
    if X_test is not None:
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader, test_loader, (X_test, y_test)
    else:
        return train_loader, val_loader, None, (None, None)


def train_with_early_stopping(model, train_loader, val_loader, device, epochs=100, patience=15):
    """
    Train model with early stopping and learning rate scheduling.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device for computation
        epochs: Maximum number of epochs
        patience: Early stopping patience
        
    Returns:
        Tuple of (best_model_state, training_history)
    """
    logger.info("Starting training with early stopping...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        history['learning_rate'].append(current_lr)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                       f"LR: {current_lr:.6f}")
        
        # Early stopping check
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    return best_model_state, history


def evaluate_model(model, test_loader, device):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device for computation
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model on test set...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    # Classification report
    class_report = classification_report(all_labels, all_predictions, 
                                        target_names=[STATE_LABELS[i] for i in range(N_CLASSES)],
                                        output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1-Score: {f1:.4f}")
    
    return results


def cross_validate_model(X, y, device, k_folds=5):
    """
    Perform k-fold cross-validation.
    
    Args:
        X: EEG data
        y: Labels
        device: Device for computation
        k_folds: Number of folds
        
    Returns:
        Dictionary with cross-validation results
    """
    logger.info(f"Performing {k_folds}-fold cross-validation...")
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_results = {
        'fold_accuracies': [],
        'fold_losses': [],
        'fold_histories': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Training fold {fold + 1}/{k_folds}")
        
        # Split data for this fold
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create data loaders for this fold
        train_loader, val_loader, _, _ = create_data_loaders(
            X_train_fold, y_train_fold, batch_size=BATCH_SIZE, test_size=None, val_size=0.2
        )
        
        # Create model for this fold
        model = CNNLSTMHybrid(N_CHANNELS, N_TIMEPOINTS, N_CLASSES).to(device)
        
        # Train model
        best_state, history = train_with_early_stopping(
            model, train_loader, val_loader, device, epochs=EPOCHS, patience=EARLY_STOPPING_PATIENCE
        )
        
        # Load best model and evaluate on validation set
        model.load_state_dict(best_state)
        val_results = evaluate_model(model, val_loader, device)
        
        cv_results['fold_accuracies'].append(val_results['accuracy'])
        cv_results['fold_losses'].append(min(history['val_loss']))
        cv_results['fold_histories'].append(history)
        
        logger.info(f"Fold {fold + 1} - Validation Accuracy: {val_results['accuracy']:.4f}")
    
    # Calculate average metrics
    cv_results['mean_accuracy'] = np.mean(cv_results['fold_accuracies'])
    cv_results['std_accuracy'] = np.std(cv_results['fold_accuracies'])
    cv_results['mean_loss'] = np.mean(cv_results['fold_losses'])
    cv_results['std_loss'] = np.std(cv_results['fold_losses'])
    
    logger.info(f"Cross-validation results:")
    logger.info(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    logger.info(f"Mean Loss: {cv_results['mean_loss']:.4f} ± {cv_results['std_loss']:.4f}")
    
    return cv_results


def plot_training_curves(history, save_path):
    """
    Plot and save training curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Training Accuracy', color='blue')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate
    axes[1, 0].plot(history['learning_rate'], color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')
    
    # Loss difference (overfitting detection)
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 1].plot(loss_diff, color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Loss - Training Loss')
    axes[1, 1].set_title('Overfitting Detection')
    axes[1, 1].grid(True)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved to: {save_path}")


def plot_confusion_matrix(cm, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    im = plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    
    # Add labels
    tick_marks = np.arange(len(STATE_LABELS))
    plt.xticks(tick_marks, [STATE_LABELS[i] for i in range(N_CLASSES)], rotation=45)
    plt.yticks(tick_marks, [STATE_LABELS[i] for i in range(N_CLASSES)])
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Counts and Normalized)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to: {save_path}")


def save_training_results(model, history, test_results, cv_results, save_dir):
    """
    Save comprehensive training results.
    
    Args:
        model: Trained model
        history: Training history
        test_results: Test evaluation results
        cv_results: Cross-validation results
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'trained_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_channels': N_CHANNELS,
            'n_timepoints': N_TIMEPOINTS,
            'n_classes': N_CLASSES
        },
        'training_history': history,
        'test_results': test_results,
        'cv_results': cv_results,
        'training_params': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'k_folds': K_FOLDS
        },
        'timestamp': datetime.now().isoformat()
    }, model_path)
    
    # Save results as JSON
    results_path = os.path.join(save_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'test_accuracy': float(test_results['accuracy']),
            'test_precision': float(test_results['precision']),
            'test_recall': float(test_results['recall']),
            'test_f1_score': float(test_results['f1_score']),
            'cv_mean_accuracy': float(cv_results['mean_accuracy']),
            'cv_std_accuracy': float(cv_results['std_accuracy']),
            'best_val_loss': float(min(history['val_loss'])),
            'final_train_accuracy': float(history['train_acc'][-1]),
            'final_val_accuracy': float(history['val_acc'][-1]),
            'training_params': {
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'learning_rate': LEARNING_RATE,
                'early_stopping_patience': EARLY_STOPPING_PATIENCE,
                'k_folds': K_FOLDS
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Results saved to: {results_path}")


def main():
    """Main training function."""
    logger.info("🚀 Starting improved EEG model training...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Generate realistic EEG data
    X, y = generate_realistic_eeg_data(n_samples=3000)
    
    # Perform cross-validation
    cv_results = cross_validate_model(X, y, device, k_folds=K_FOLDS)
    
    # Create final train/test splits
    train_loader, val_loader, test_loader, (X_test, y_test) = create_data_loaders(
        X, y, batch_size=BATCH_SIZE, test_size=0.2, val_size=0.2
    )
    
    # Train final model
    logger.info("Training final model...")
    model = CNNLSTMHybrid(N_CHANNELS, N_TIMEPOINTS, N_CLASSES).to(device)
    best_state, history = train_with_early_stopping(
        model, train_loader, val_loader, device, epochs=EPOCHS, patience=EARLY_STOPPING_PATIENCE
    )
    
    # Load best model and evaluate
    model.load_state_dict(best_state)
    test_results = evaluate_model(model, test_loader, device)
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save results
    save_training_results(model, history, test_results, cv_results, save_dir)
    
    # Plot and save visualizations
    plot_training_curves(history, os.path.join(save_dir, 'training_curves.png'))
    plot_confusion_matrix(np.array(test_results['confusion_matrix']), 
                         os.path.join(save_dir, 'confusion_matrix.png'))
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("🎯 TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Final Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Cross-validation Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    logger.info(f"Test Precision: {test_results['precision']:.4f}")
    logger.info(f"Test Recall: {test_results['recall']:.4f}")
    logger.info(f"Test F1-Score: {test_results['f1_score']:.4f}")
    logger.info(f"Best Validation Loss: {min(history['val_loss']):.4f}")
    logger.info("="*60)
    
    logger.info("✅ Training completed successfully!")
    logger.info(f"📁 All results saved to: {save_dir}")
    
    return model, test_results, cv_results


if __name__ == "__main__":
    trained_model, results, cv_results = main()
