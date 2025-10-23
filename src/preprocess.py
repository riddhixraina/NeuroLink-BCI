"""
EEG Preprocessing Pipeline for NeuroLink-BCI
Implements band-pass filtering, artifact removal, and signal segmentation.
"""

import numpy as np
import mne
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """
    Comprehensive EEG preprocessing pipeline including filtering, artifact removal, and segmentation.
    """
    
    def __init__(self, sampling_rate: int = 128, n_channels: int = 32):
        """
        Initialize the EEG preprocessor.
        
        Args:
            sampling_rate: EEG sampling rate in Hz
            n_channels: Number of EEG channels
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.filter_params = {
            'low_freq': 0.5,
            'high_freq': 45.0,
            'notch_freq': 50.0  # For power line noise
        }
        
    def apply_bandpass_filter(self, eeg_data: np.ndarray, 
                             low_freq: float = 0.5, 
                             high_freq: float = 45.0) -> np.ndarray:
        """
        Apply band-pass filter to EEG data.
        
        Args:
            eeg_data: EEG data (trials, channels, time_points)
            low_freq: Low cutoff frequency
            high_freq: High cutoff frequency
            
        Returns:
            Filtered EEG data
        """
        logger.info(f"Applying band-pass filter ({low_freq}-{high_freq} Hz)...")
        
        # Design Butterworth filter
        nyquist = self.sampling_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        b, a = butter(4, [low_norm, high_norm], btype='band')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(eeg_data)
        
        for trial in range(eeg_data.shape[0]):
            for channel in range(eeg_data.shape[1]):
                filtered_data[trial, channel, :] = filtfilt(b, a, eeg_data[trial, channel, :])
        
        logger.info("Band-pass filtering completed.")
        return filtered_data
    
    def apply_notch_filter(self, eeg_data: np.ndarray, 
                          notch_freq: float = 50.0) -> np.ndarray:
        """
        Apply notch filter to remove power line noise.
        
        Args:
            eeg_data: EEG data (trials, channels, time_points)
            notch_freq: Notch frequency (usually 50 or 60 Hz)
            
        Returns:
            Filtered EEG data
        """
        logger.info(f"Applying notch filter at {notch_freq} Hz...")
        
        # Design notch filter
        nyquist = self.sampling_rate / 2
        notch_norm = notch_freq / nyquist
        
        b, a = butter(4, [notch_norm - 0.01, notch_norm + 0.01], btype='bandstop')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(eeg_data)
        
        for trial in range(eeg_data.shape[0]):
            for channel in range(eeg_data.shape[1]):
                filtered_data[trial, channel, :] = filtfilt(b, a, eeg_data[trial, channel, :])
        
        logger.info("Notch filtering completed.")
        return filtered_data
    
    def detect_artifacts(self, eeg_data: np.ndarray, 
                        threshold_std: float = 3.0) -> np.ndarray:
        """
        Detect artifacts using statistical methods.
        
        Args:
            eeg_data: EEG data (trials, channels, time_points)
            threshold_std: Standard deviation threshold for artifact detection
            
        Returns:
            Boolean array indicating artifact presence
        """
        logger.info("Detecting artifacts...")
        
        artifacts = np.zeros(eeg_data.shape[0], dtype=bool)
        
        for trial in range(eeg_data.shape[0]):
            # Check for amplitude outliers
            trial_data = eeg_data[trial, :, :]
            mean_amp = np.mean(np.abs(trial_data))
            std_amp = np.std(trial_data)
            
            if mean_amp > threshold_std * std_amp:
                artifacts[trial] = True
                continue
            
            # Check for flat channels
            for channel in range(trial_data.shape[0]):
                if np.std(trial_data[channel, :]) < 1e-6:
                    artifacts[trial] = True
                    break
        
        logger.info(f"Detected {np.sum(artifacts)} artifacts out of {len(artifacts)} trials.")
        return artifacts
    
    def remove_artifacts(self, eeg_data: np.ndarray, 
                        labels: np.ndarray,
                        artifact_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove trials with artifacts.
        
        Args:
            eeg_data: EEG data (trials, channels, time_points)
            labels: Trial labels
            artifact_mask: Boolean array indicating artifacts
            
        Returns:
            Clean EEG data and corresponding labels
        """
        logger.info("Removing artifacts...")
        
        clean_mask = ~artifact_mask
        clean_data = eeg_data[clean_mask]
        clean_labels = labels[clean_mask]
        
        logger.info(f"Removed {np.sum(artifact_mask)} trials. "
                   f"Remaining: {len(clean_data)} trials.")
        
        return clean_data, clean_labels
    
    def segment_data(self, eeg_data: np.ndarray, 
                    window_size: float = 2.0,
                    overlap: float = 0.5) -> np.ndarray:
        """
        Segment EEG data into fixed-length windows.
        
        Args:
            eeg_data: EEG data (trials, channels, time_points)
            window_size: Window size in seconds
            overlap: Overlap ratio (0-1)
            
        Returns:
            Segmented EEG data
        """
        logger.info(f"Segmenting data into {window_size}s windows with {overlap*100}% overlap...")
        
        window_samples = int(window_size * self.sampling_rate)
        step_size = int(window_samples * (1 - overlap))
        
        segmented_data = []
        
        for trial in range(eeg_data.shape[0]):
            trial_data = eeg_data[trial, :, :]
            n_samples = trial_data.shape[1]
            
            # Create overlapping windows
            start_idx = 0
            while start_idx + window_samples <= n_samples:
                window = trial_data[:, start_idx:start_idx + window_samples]
                segmented_data.append(window)
                start_idx += step_size
        
        segmented_data = np.array(segmented_data)
        logger.info(f"Created {segmented_data.shape[0]} segments of shape {segmented_data.shape[1:]}")
        
        return segmented_data
    
    def normalize_data(self, eeg_data: np.ndarray, 
                      method: str = 'z_score') -> np.ndarray:
        """
        Normalize EEG data.
        
        Args:
            eeg_data: EEG data (segments, channels, time_points)
            method: Normalization method ('z_score', 'minmax', 'robust')
            
        Returns:
            Normalized EEG data
        """
        logger.info(f"Normalizing data using {method} method...")
        
        normalized_data = np.zeros_like(eeg_data)
        
        if method == 'z_score':
            scaler = StandardScaler()
            for segment in range(eeg_data.shape[0]):
                segment_data = eeg_data[segment, :, :].T  # (time, channels)
                normalized_segment = scaler.fit_transform(segment_data).T  # (channels, time)
                normalized_data[segment, :, :] = normalized_segment
                
        elif method == 'minmax':
            for segment in range(eeg_data.shape[0]):
                segment_data = eeg_data[segment, :, :]
                min_val = np.min(segment_data)
                max_val = np.max(segment_data)
                normalized_data[segment, :, :] = (segment_data - min_val) / (max_val - min_val)
                
        elif method == 'robust':
            for segment in range(eeg_data.shape[0]):
                segment_data = eeg_data[segment, :, :]
                median = np.median(segment_data)
                mad = np.median(np.abs(segment_data - median))
                normalized_data[segment, :, :] = (segment_data - median) / mad
        
        logger.info("Data normalization completed.")
        return normalized_data
    
    def apply_ica_artifact_removal(self, eeg_data: np.ndarray, 
                                  n_components: int = 0.99) -> np.ndarray:
        """
        Apply Independent Component Analysis for artifact removal.
        
        Args:
            eeg_data: EEG data (trials, channels, time_points)
            n_components: Number of ICA components (or explained variance ratio)
            
        Returns:
            EEG data with artifacts removed via ICA
        """
        logger.info("Applying ICA artifact removal...")
        
        # Flatten data for ICA (concatenate all trials)
        n_trials, n_channels, n_time = eeg_data.shape
        data_flat = eeg_data.reshape(-1, n_channels).T  # (channels, all_time)
        
        # Create MNE Raw object for ICA
        info = mne.create_info(ch_names=[f'Ch{i}' for i in range(n_channels)],
                              sfreq=self.sampling_rate,
                              ch_types=['eeg'] * n_channels)
        raw = mne.io.RawArray(data_flat, info)
        
        # Apply ICA
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
        ica.fit(raw)
        
        # Automatic artifact detection and removal
        ica.exclude = []
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
        
        ica.exclude = eog_indices + ecg_indices
        
        # Apply ICA
        ica.apply(raw)
        
        # Convert back to original shape
        cleaned_data = raw.get_data().T.reshape(n_trials, n_channels, n_time)
        
        logger.info(f"ICA completed. Removed {len(ica.exclude)} components.")
        return cleaned_data
    
    def preprocess_pipeline(self, eeg_data: np.ndarray, 
                           labels: np.ndarray,
                           apply_ica: bool = True,
                           segment_data: bool = True) -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            eeg_data: Raw EEG data (trials, channels, time_points)
            labels: Trial labels
            apply_ica: Whether to apply ICA artifact removal
            segment_data: Whether to segment data into windows
            
        Returns:
            Dictionary containing preprocessed data and metadata
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # Step 1: Band-pass filtering
        filtered_data = self.apply_bandpass_filter(eeg_data)
        
        # Step 2: Notch filtering
        filtered_data = self.apply_notch_filter(filtered_data)
        
        # Step 3: Artifact detection and removal
        artifacts = self.detect_artifacts(filtered_data)
        clean_data, clean_labels = self.remove_artifacts(filtered_data, labels, artifacts)
        
        # Step 4: ICA artifact removal (optional)
        if apply_ica and clean_data.shape[0] > 0:
            clean_data = self.apply_ica_artifact_removal(clean_data)
        
        # Step 5: Data segmentation (optional)
        if segment_data and clean_data.shape[0] > 0:
            clean_data = self.segment_data(clean_data)
            # Replicate labels for each segment
            n_segments_per_trial = clean_data.shape[0] // len(clean_labels)
            clean_labels = np.repeat(clean_labels, n_segments_per_trial)
        
        # Step 6: Normalization
        if clean_data.shape[0] > 0:
            clean_data = self.normalize_data(clean_data)
        
        # Create results dictionary
        results = {
            'eeg_data': clean_data,
            'labels': clean_labels,
            'sampling_rate': self.sampling_rate,
            'n_channels': self.n_channels,
            'n_trials_original': eeg_data.shape[0],
            'n_trials_clean': clean_data.shape[0],
            'artifacts_removed': np.sum(artifacts),
            'preprocessing_info': {
                'filtered': True,
                'ica_applied': apply_ica,
                'segmented': segment_data,
                'normalized': True
            }
        }
        
        logger.info("Preprocessing pipeline completed successfully.")
        logger.info(f"Final data shape: {clean_data.shape}")
        
        return results


def main():
    """Example usage of the EEGPreprocessor."""
    # Create mock EEG data
    n_trials = 50
    n_channels = 32
    n_time = 7680  # 60 seconds at 128 Hz
    sampling_rate = 128
    
    # Generate mock EEG data with some artifacts
    eeg_data = np.random.randn(n_trials, n_channels, n_time) * 50
    
    # Add some artifacts
    eeg_data[5, :, :] *= 5  # Amplitude artifact
    eeg_data[10, 5, :] = 0  # Flat channel artifact
    
    # Generate mock labels
    labels = np.random.randint(0, 3, n_trials)
    
    # Initialize preprocessor
    preprocessor = EEGPreprocessor(sampling_rate=sampling_rate, n_channels=n_channels)
    
    # Run preprocessing pipeline
    results = preprocessor.preprocess_pipeline(eeg_data, labels)
    
    print("Preprocessing Results:")
    print(f"Original trials: {results['n_trials_original']}")
    print(f"Clean trials: {results['n_trials_clean']}")
    print(f"Artifacts removed: {results['artifacts_removed']}")
    print(f"Final data shape: {results['eeg_data'].shape}")
    print(f"Final labels shape: {results['labels'].shape}")


if __name__ == "__main__":
    main()
