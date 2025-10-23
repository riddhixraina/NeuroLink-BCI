"""
Feature Extraction for EEG Neural Decoding
Implements frequency-domain, time-frequency, and connectivity-based features.
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from scipy.signal import welch, coherence
import pywt
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGFeatureExtractor:
    """
    Comprehensive feature extraction for EEG signals including frequency, time-frequency, and connectivity features.
    """
    
    def __init__(self, sampling_rate: int = 128):
        """
        Initialize the feature extractor.
        
        Args:
            sampling_rate: EEG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
    def extract_power_spectral_density(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract Power Spectral Density (PSD) features for each frequency band.
        
        Args:
            eeg_data: EEG data (segments, channels, time_points)
            
        Returns:
            Dictionary containing PSD features for each frequency band
        """
        logger.info("Extracting Power Spectral Density features...")
        
        n_segments, n_channels, n_time = eeg_data.shape
        psd_features = {}
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            band_psd = np.zeros((n_segments, n_channels))
            
            for segment in range(n_segments):
                for channel in range(n_channels):
                    # Compute PSD using Welch's method
                    freqs, psd = welch(eeg_data[segment, channel, :], 
                                     fs=self.sampling_rate, 
                                     nperseg=min(256, n_time//4))
                    
                    # Find frequency indices for the band
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    
                    # Compute average power in the band
                    band_psd[segment, channel] = np.mean(psd[band_mask])
            
            psd_features[band_name] = band_psd
        
        logger.info("PSD feature extraction completed.")
        return psd_features
    
    def extract_frequency_ratios(self, psd_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract frequency band ratios for attention and stress metrics.
        
        Args:
            psd_features: Dictionary containing PSD features for each band
            
        Returns:
            Dictionary containing frequency ratios
        """
        logger.info("Extracting frequency ratios...")
        
        ratios = {}
        
        # Theta/Beta ratio (attention metric)
        if 'theta' in psd_features and 'beta' in psd_features:
            ratios['theta_beta'] = psd_features['theta'] / (psd_features['beta'] + 1e-8)
        
        # Alpha/Beta ratio (relaxation metric)
        if 'alpha' in psd_features and 'beta' in psd_features:
            ratios['alpha_beta'] = psd_features['alpha'] / (psd_features['beta'] + 1e-8)
        
        # Gamma/Alpha ratio (cognitive load metric)
        if 'gamma' in psd_features and 'alpha' in psd_features:
            ratios['gamma_alpha'] = psd_features['gamma'] / (psd_features['alpha'] + 1e-8)
        
        # Delta/Alpha ratio (arousal metric)
        if 'delta' in psd_features and 'alpha' in psd_features:
            ratios['delta_alpha'] = psd_features['delta'] / (psd_features['alpha'] + 1e-8)
        
        logger.info("Frequency ratio extraction completed.")
        return ratios
    
    def extract_wavelet_features(self, eeg_data: np.ndarray, 
                               wavelet: str = 'db4') -> Dict[str, np.ndarray]:
        """
        Extract time-frequency features using wavelet transform.
        
        Args:
            eeg_data: EEG data (segments, channels, time_points)
            wavelet: Wavelet type for decomposition
            
        Returns:
            Dictionary containing wavelet features
        """
        logger.info("Extracting wavelet features...")
        
        n_segments, n_channels, n_time = eeg_data.shape
        wavelet_features = {}
        
        # Define decomposition levels
        max_level = int(np.log2(n_time))
        levels = min(max_level, 6)  # Limit to 6 levels
        
        for level in range(1, levels + 1):
            band_name = f'level_{level}'
            level_features = np.zeros((n_segments, n_channels))
            
            for segment in range(n_segments):
                for channel in range(n_channels):
                    # Perform wavelet decomposition
                    coeffs = pywt.wavedec(eeg_data[segment, channel, :], 
                                        wavelet, 
                                        level=level)
                    
                    # Use detail coefficients (high frequency components)
                    if level <= len(coeffs) - 1:
                        detail_coeffs = coeffs[-level]
                        level_features[segment, channel] = np.var(detail_coeffs)
            
            wavelet_features[band_name] = level_features
        
        logger.info("Wavelet feature extraction completed.")
        return wavelet_features
    
    def extract_connectivity_features(self, eeg_data: np.ndarray, 
                                    method: str = 'coherence') -> Dict[str, np.ndarray]:
        """
        Extract connectivity features between electrode pairs.
        
        Args:
            eeg_data: EEG data (segments, channels, time_points)
            method: Connectivity method ('coherence', 'plv', 'pli')
            
        Returns:
            Dictionary containing connectivity features
        """
        logger.info(f"Extracting connectivity features using {method}...")
        
        n_segments, n_channels, n_time = eeg_data.shape
        
        if method == 'coherence':
            return self._extract_coherence_features(eeg_data)
        elif method == 'plv':
            return self._extract_plv_features(eeg_data)
        elif method == 'pli':
            return self._extract_pli_features(eeg_data)
        else:
            raise ValueError(f"Unknown connectivity method: {method}")
    
    def _extract_coherence_features(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract coherence-based connectivity features."""
        n_segments, n_channels, n_time = eeg_data.shape
        coherence_features = {}
        
        # Select representative electrode pairs for efficiency
        electrode_pairs = [(i, j) for i in range(0, n_channels, 2) 
                          for j in range(i+2, n_channels, 2)]
        
        for freq_band, (low_freq, high_freq) in self.frequency_bands.items():
            band_coherence = np.zeros((n_segments, len(electrode_pairs)))
            
            for segment in range(n_segments):
                for pair_idx, (ch1, ch2) in enumerate(electrode_pairs):
                    # Compute coherence
                    freqs, coh = coherence(eeg_data[segment, ch1, :], 
                                         eeg_data[segment, ch2, :], 
                                         fs=self.sampling_rate)
                    
                    # Average coherence in the frequency band
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    band_coherence[segment, pair_idx] = np.mean(coh[band_mask])
            
            coherence_features[f'{freq_band}_coherence'] = band_coherence
        
        return coherence_features
    
    def _extract_plv_features(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Phase Locking Value (PLV) features."""
        n_segments, n_channels, n_time = eeg_data.shape
        plv_features = {}
        
        # Select representative electrode pairs
        electrode_pairs = [(i, j) for i in range(0, n_channels, 2) 
                          for j in range(i+2, n_channels, 2)]
        
        for freq_band, (low_freq, high_freq) in self.frequency_bands.items():
            band_plv = np.zeros((n_segments, len(electrode_pairs)))
            
            for segment in range(n_segments):
                for pair_idx, (ch1, ch2) in enumerate(electrode_pairs):
                    # Band-pass filter
                    sos = signal.butter(4, [low_freq, high_freq], 
                                      btype='band', fs=self.sampling_rate, output='sos')
                    filtered_ch1 = signal.sosfilt(sos, eeg_data[segment, ch1, :])
                    filtered_ch2 = signal.sosfilt(sos, eeg_data[segment, ch2, :])
                    
                    # Extract instantaneous phase
                    phase1 = np.angle(signal.hilbert(filtered_ch1))
                    phase2 = np.angle(signal.hilbert(filtered_ch2))
                    
                    # Compute PLV
                    phase_diff = phase1 - phase2
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    band_plv[segment, pair_idx] = plv
            
            plv_features[f'{freq_band}_plv'] = band_plv
        
        return plv_features
    
    def _extract_pli_features(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Phase Lag Index (PLI) features."""
        n_segments, n_channels, n_time = eeg_data.shape
        pli_features = {}
        
        # Select representative electrode pairs
        electrode_pairs = [(i, j) for i in range(0, n_channels, 2) 
                          for j in range(i+2, n_channels, 2)]
        
        for freq_band, (low_freq, high_freq) in self.frequency_bands.items():
            band_pli = np.zeros((n_segments, len(electrode_pairs)))
            
            for segment in range(n_segments):
                for pair_idx, (ch1, ch2) in enumerate(electrode_pairs):
                    # Band-pass filter
                    sos = signal.butter(4, [low_freq, high_freq], 
                                      btype='band', fs=self.sampling_rate, output='sos')
                    filtered_ch1 = signal.sosfilt(sos, eeg_data[segment, ch1, :])
                    filtered_ch2 = signal.sosfilt(sos, eeg_data[segment, ch2, :])
                    
                    # Extract instantaneous phase
                    phase1 = np.angle(signal.hilbert(filtered_ch1))
                    phase2 = np.angle(signal.hilbert(filtered_ch2))
                    
                    # Compute PLI
                    phase_diff = phase1 - phase2
                    pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                    band_pli[segment, pair_idx] = pli
            
            pli_features[f'{freq_band}_pli'] = band_pli
        
        return pli_features
    
    def extract_statistical_features(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract statistical features from EEG signals.
        
        Args:
            eeg_data: EEG data (segments, channels, time_points)
            
        Returns:
            Dictionary containing statistical features
        """
        logger.info("Extracting statistical features...")
        
        n_segments, n_channels, n_time = eeg_data.shape
        stat_features = {}
        
        # Mean amplitude
        stat_features['mean_amplitude'] = np.mean(np.abs(eeg_data), axis=2)
        
        # Standard deviation
        stat_features['std_amplitude'] = np.std(eeg_data, axis=2)
        
        # Skewness
        stat_features['skewness'] = skew(eeg_data, axis=2)
        
        # Kurtosis
        stat_features['kurtosis'] = kurtosis(eeg_data, axis=2)
        
        # Peak-to-peak amplitude
        stat_features['peak_to_peak'] = np.ptp(eeg_data, axis=2)
        
        # Zero crossings
        zero_crossings = np.zeros((n_segments, n_channels))
        for segment in range(n_segments):
            for channel in range(n_channels):
                signal_data = eeg_data[segment, channel, :]
                zero_crossings[segment, channel] = np.sum(np.diff(np.sign(signal_data)) != 0)
        stat_features['zero_crossings'] = zero_crossings
        
        logger.info("Statistical feature extraction completed.")
        return stat_features
    
    def extract_novelty_features(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract novelty detection features inspired by hippocampal circuits.
        
        Args:
            eeg_data: EEG data (segments, channels, time_points)
            
        Returns:
            Dictionary containing novelty features
        """
        logger.info("Extracting novelty features...")
        
        n_segments, n_channels, n_time = eeg_data.shape
        novelty_features = {}
        
        # Compute novelty index based on signal variance changes
        novelty_index = np.zeros((n_segments, n_channels))
        
        for channel in range(n_channels):
            channel_data = eeg_data[:, channel, :]
            
            # Compute sliding window variance
            window_size = min(100, n_time // 4)
            variances = []
            
            for segment in range(n_segments):
                segment_data = channel_data[segment, :]
                
                # Compute variance in sliding windows
                segment_variances = []
                for i in range(0, len(segment_data) - window_size, window_size // 2):
                    window = segment_data[i:i + window_size]
                    segment_variances.append(np.var(window))
                
                # Novelty as variance of variances (surprise measure)
                novelty_index[segment, channel] = np.var(segment_variances)
        
        novelty_features['novelty_index'] = novelty_index
        
        # Compute pattern complexity using entropy
        complexity_features = np.zeros((n_segments, n_channels))
        
        for segment in range(n_segments):
            for channel in range(n_channels):
                signal_data = eeg_data[segment, channel, :]
                
                # Discretize signal into bins
                hist, _ = np.histogram(signal_data, bins=20)
                hist = hist / np.sum(hist)  # Normalize
                hist = hist[hist > 0]  # Remove zero bins
                
                # Compute Shannon entropy
                entropy = -np.sum(hist * np.log2(hist))
                complexity_features[segment, channel] = entropy
        
        novelty_features['pattern_complexity'] = complexity_features
        
        logger.info("Novelty feature extraction completed.")
        return novelty_features
    
    def extract_all_features(self, eeg_data: np.ndarray, 
                           connectivity_method: str = 'coherence') -> Dict[str, np.ndarray]:
        """
        Extract all available features from EEG data.
        
        Args:
            eeg_data: EEG data (segments, channels, time_points)
            connectivity_method: Method for connectivity features
            
        Returns:
            Dictionary containing all extracted features
        """
        logger.info("Extracting all features...")
        
        all_features = {}
        
        # PSD features
        psd_features = self.extract_power_spectral_density(eeg_data)
        all_features.update(psd_features)
        
        # Frequency ratios
        ratio_features = self.extract_frequency_ratios(psd_features)
        all_features.update(ratio_features)
        
        # Wavelet features
        wavelet_features = self.extract_wavelet_features(eeg_data)
        all_features.update(wavelet_features)
        
        # Connectivity features
        connectivity_features = self.extract_connectivity_features(eeg_data, connectivity_method)
        all_features.update(connectivity_features)
        
        # Statistical features
        stat_features = self.extract_statistical_features(eeg_data)
        all_features.update(stat_features)
        
        # Novelty features
        novelty_features = self.extract_novelty_features(eeg_data)
        all_features.update(novelty_features)
        
        logger.info(f"Feature extraction completed. Total features: {len(all_features)}")
        
        return all_features
    
    def flatten_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Flatten all features into a single feature matrix.
        
        Args:
            features: Dictionary containing feature arrays
            
        Returns:
            Flattened feature matrix (samples, features)
        """
        logger.info("Flattening features...")
        
        feature_matrices = []
        feature_names = []
        
        for name, feature_array in features.items():
            # Flatten each feature array
            if feature_array.ndim == 2:  # (segments, channels)
                flattened = feature_array.reshape(feature_array.shape[0], -1)
                feature_matrices.append(flattened)
                
                # Generate feature names
                n_channels = feature_array.shape[1]
                for i in range(n_channels):
                    feature_names.append(f"{name}_ch{i}")
            else:
                feature_matrices.append(feature_array)
                feature_names.append(name)
        
        # Concatenate all features
        flattened_features = np.concatenate(feature_matrices, axis=1)
        
        logger.info(f"Features flattened to shape: {flattened_features.shape}")
        
        return flattened_features, feature_names


def main():
    """Example usage of the EEGFeatureExtractor."""
    # Create mock EEG data
    n_segments = 100
    n_channels = 32
    n_time = 256  # 2 seconds at 128 Hz
    sampling_rate = 128
    
    # Generate mock EEG data
    eeg_data = np.random.randn(n_segments, n_channels, n_time) * 50
    
    # Add some frequency content
    t = np.linspace(0, 2, n_time)
    for i in range(n_segments):
        for j in range(n_channels):
            eeg_data[i, j, :] += 10 * np.sin(2 * np.pi * 10 * t)  # Alpha waves
            eeg_data[i, j, :] += 5 * np.sin(2 * np.pi * 20 * t)   # Beta waves
    
    # Initialize feature extractor
    extractor = EEGFeatureExtractor(sampling_rate=sampling_rate)
    
    # Extract all features
    features = extractor.extract_all_features(eeg_data)
    
    print("Feature Extraction Results:")
    print(f"Number of feature types: {len(features)}")
    for name, feature_array in features.items():
        print(f"{name}: {feature_array.shape}")
    
    # Flatten features
    flattened_features, feature_names = extractor.flatten_features(features)
    print(f"\nFlattened features shape: {flattened_features.shape}")
    print(f"Number of feature names: {len(feature_names)}")


if __name__ == "__main__":
    main()
