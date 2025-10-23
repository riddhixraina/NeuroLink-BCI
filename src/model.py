"""
CNN-LSTM Hybrid Model for EEG Neural State Classification
Implements spatial and temporal feature extraction for cognitive state prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG data.
    """
    
    def __init__(self, eeg_data: np.ndarray, labels: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            eeg_data: EEG data (samples, channels, time_points)
            labels: Corresponding labels
        """
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]


class SpatialCNN(nn.Module):
    """
    Spatial CNN for extracting features across EEG electrodes.
    """
    
    def __init__(self, n_channels: int, n_filters: int = 64):
        """
        Initialize the spatial CNN.
        
        Args:
            n_channels: Number of EEG channels
            n_filters: Number of filters in the first convolutional layer
        """
        super(SpatialCNN, self).__init__()
        
        self.n_channels = n_channels
        self.n_filters = n_filters
        
        # Spatial convolution layers
        self.conv1 = nn.Conv2d(1, n_filters, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, kernel_size=(1, 3), padding=(0, 1))
        self.conv3 = nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=(1, 3), padding=(0, 1))
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.bn2 = nn.BatchNorm2d(n_filters * 2)
        self.bn3 = nn.BatchNorm2d(n_filters * 4)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        """
        Forward pass through spatial CNN.
        
        Args:
            x: Input tensor (batch_size, 1, channels, time_points)
            
        Returns:
            Spatial features (batch_size, n_filters * 4)
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch_size, 1, channels, time_points)
        
        # Spatial convolution layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        return x


class TemporalLSTM(nn.Module):
    """
    Temporal LSTM for capturing temporal patterns in EEG signals.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        """
        Initialize the temporal LSTM.
        
        Args:
            input_size: Input feature size
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
        """
        super(TemporalLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, dropout=0.1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
    def forward(self, x):
        """
        Forward pass through temporal LSTM.
        
        Args:
            x: Input tensor (batch_size, channels, time_points)
            
        Returns:
            Temporal features (batch_size, hidden_size * 2)
        """
        # Transpose to (batch_size, time_points, channels)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention mechanism
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Layer normalization
        attended_out = self.layer_norm(attended_out)
        
        # Global average pooling over time
        temporal_features = torch.mean(attended_out, dim=1)
        
        return temporal_features


class CNNLSTMHybrid(nn.Module):
    """
    Hybrid CNN-LSTM model for EEG neural state classification.
    """
    
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int, 
                 spatial_filters: int = 64, lstm_hidden: int = 128, 
                 lstm_layers: int = 2):
        """
        Initialize the hybrid model.
        
        Args:
            n_channels: Number of EEG channels
            n_timepoints: Number of time points per sample
            n_classes: Number of output classes
            spatial_filters: Number of filters in spatial CNN
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
        """
        super(CNNLSTMHybrid, self).__init__()
        
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes
        
        # Spatial CNN branch
        self.spatial_cnn = SpatialCNN(n_channels, spatial_filters)
        
        # Temporal LSTM branch
        self.temporal_lstm = TemporalLSTM(n_channels, lstm_hidden, lstm_layers)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(spatial_filters * 4 + lstm_hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor (batch_size, channels, time_points)
            
        Returns:
            Classification logits (batch_size, n_classes)
        """
        # Spatial feature extraction
        spatial_features = self.spatial_cnn(x)
        
        # Temporal feature extraction
        temporal_features = self.temporal_lstm(x)
        
        # Feature fusion
        combined_features = torch.cat([spatial_features, temporal_features], dim=1)
        
        # Classification
        output = self.fusion(combined_features)
        
        return output
    
    def get_attention_weights(self, x):
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input tensor (batch_size, channels, time_points)
            
        Returns:
            Attention weights
        """
        # Get temporal features and attention weights
        x_transposed = x.transpose(1, 2)
        lstm_out, _ = self.temporal_lstm.lstm(x_transposed)
        attention_out, attention_weights = self.temporal_lstm.attention(lstm_out, lstm_out, lstm_out)
        
        return attention_weights


class EEGClassifier:
    """
    Main classifier class for training and evaluating the CNN-LSTM model.
    """
    
    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the classifier.
        
        Args:
            n_channels: Number of EEG channels
            n_timepoints: Number of time points per sample
            n_classes: Number of output classes
            device: Device for computation ('cuda' or 'cpu')
        """
        self.device = device
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes
        
        # Initialize model
        self.model = CNNLSTMHybrid(n_channels, n_timepoints, n_classes).to(device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                   mode='min', 
                                                                   patience=10, 
                                                                   factor=0.5)
        
        # Training history
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, patience: int = 20) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        logger.info("Starting training...")
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                           f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, "
                           f"Val Acc: {val_acc:.2f}%")
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_accuracy': best_val_acc
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating model...")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100. * np.mean(np.array(all_predictions) == np.array(all_targets))
        
        # Classification report
        report = classification_report(all_targets, all_predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        logger.info(f"Test accuracy: {accuracy:.2f}%")
        
        return results
    
    def predict(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new EEG data.
        
        Args:
            eeg_data: EEG data (samples, channels, time_points)
            
        Returns:
            Predictions and confidence scores
        """
        self.model.eval()
        
        # Convert to tensor
        eeg_tensor = torch.FloatTensor(eeg_data).to(self.device)
        
        with torch.no_grad():
            output = self.model(eeg_tensor)
            probabilities = F.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            confidence = probabilities.max(dim=1)[0]
        
        return predictions.cpu().numpy(), confidence.cpu().numpy()


def main():
    """Example usage of the EEGClassifier."""
    # Create mock EEG data
    n_samples = 1000
    n_channels = 32
    n_timepoints = 256
    n_classes = 3
    
    # Generate mock data
    X = np.random.randn(n_samples, n_channels, n_timepoints)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize classifier
    classifier = EEGClassifier(n_channels, n_timepoints, n_classes)
    
    # Train model
    history = classifier.train(train_loader, val_loader, epochs=50)
    
    # Evaluate model
    results = classifier.evaluate(test_loader)
    
    print("Training Results:")
    print(f"Best validation accuracy: {history['best_val_accuracy']:.2f}%")
    print(f"Test accuracy: {results['accuracy']:.2f}%")
    
    # Make predictions on new data
    new_data = np.random.randn(10, n_channels, n_timepoints)
    predictions, confidence = classifier.predict(new_data)
    
    print(f"\nPredictions on new data:")
    print(f"Predictions: {predictions}")
    print(f"Confidence: {confidence}")


if __name__ == "__main__":
    main()
