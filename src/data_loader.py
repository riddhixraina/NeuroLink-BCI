"""
EEG Dataset Loader for NeuroLink-BCI
Handles loading and preprocessing of multiple EEG datasets including DEAP, SEED, and PhysioNet.
"""

import os
import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat
import mne
from typing import Dict, List, Tuple, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGDataLoader:
    """
    Unified data loader for multiple EEG datasets.
    Supports DEAP, SEED, and PhysioNet formats.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the directory containing EEG datasets
        """
        self.data_dir = data_dir
        self.supported_datasets = ['deap', 'seed', 'physionet']
        
    def load_deap_dataset(self, subject_id: Optional[int] = None) -> Dict:
        """
        Load DEAP dataset for emotion recognition.
        
        Args:
            subject_id: Specific subject to load (None for all subjects)
            
        Returns:
            Dictionary containing EEG data, labels, and metadata
        """
        logger.info("Loading DEAP dataset...")
        
        # DEAP dataset structure
        deap_dir = os.path.join(self.data_dir, 'deap')
        if not os.path.exists(deap_dir):
            logger.warning(f"DEAP directory not found: {deap_dir}")
            return self._create_mock_data('deap')
        
        data = {
            'eeg_data': [],
            'labels': [],
            'sampling_rate': 128,
            'channels': 32,
            'subjects': []
        }
        
        # DEAP channels (32 EEG + 8 peripheral)
        eeg_channels = [
            'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'CP5',
            'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2',
            'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'CP6',
            'CP2', 'P4', 'P8', 'PO4', 'O2', 'Fc5', 'Fc1', 'Cz'
        ]
        
        # Mock data generation for demonstration
        if subject_id is None:
            subject_ids = range(1, 33)  # 32 subjects in DEAP
        else:
            subject_ids = [subject_id]
            
        for subj in subject_ids:
            # Generate mock EEG data (40 trials x 60 seconds x 32 channels)
            mock_eeg = np.random.randn(40, 32, 7680) * 50 + np.sin(np.linspace(0, 20*np.pi, 7680)) * 10
            
            # Generate mock labels (valence, arousal, dominance, liking)
            mock_labels = np.random.randint(1, 10, (40, 4))
            
            data['eeg_data'].append(mock_eeg)
            data['labels'].append(mock_labels)
            data['subjects'].append(f'deap_subject_{subj:02d}')
        
        data['eeg_data'] = np.concatenate(data['eeg_data'], axis=0)
        data['labels'] = np.concatenate(data['labels'], axis=0)
        data['channel_names'] = eeg_channels
        
        logger.info(f"Loaded DEAP data: {data['eeg_data'].shape[0]} trials, "
                   f"{data['eeg_data'].shape[1]} channels, "
                   f"{data['eeg_data'].shape[2]} time points")
        
        return data
    
    def load_seed_dataset(self, subject_id: Optional[int] = None) -> Dict:
        """
        Load SEED dataset for affective state analysis.
        
        Args:
            subject_id: Specific subject to load (None for all subjects)
            
        Returns:
            Dictionary containing EEG data, labels, and metadata
        """
        logger.info("Loading SEED dataset...")
        
        seed_dir = os.path.join(self.data_dir, 'seed')
        if not os.path.exists(seed_dir):
            logger.warning(f"SEED directory not found: {seed_dir}")
            return self._create_mock_data('seed')
        
        data = {
            'eeg_data': [],
            'labels': [],
            'sampling_rate': 200,
            'channels': 62,
            'subjects': []
        }
        
        # SEED channels (62 EEG channels)
        eeg_channels = [
            'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
            'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
            'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
            'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
        ]
        
        # Mock data generation
        if subject_id is None:
            subject_ids = range(1, 16)  # 15 subjects in SEED
        else:
            subject_ids = [subject_id]
            
        for subj in subject_ids:
            # Generate mock EEG data (15 trials x 62 channels x 120 seconds)
            mock_eeg = np.random.randn(15, 62, 24000) * 30 + np.cos(np.linspace(0, 30*np.pi, 24000)) * 15
            
            # Generate mock labels (positive, neutral, negative emotions)
            mock_labels = np.random.choice([0, 1, 2], 15)  # 0: negative, 1: neutral, 2: positive
            
            data['eeg_data'].append(mock_eeg)
            data['labels'].append(mock_labels)
            data['subjects'].append(f'seed_subject_{subj:02d}')
        
        data['eeg_data'] = np.concatenate(data['eeg_data'], axis=0)
        data['labels'] = np.concatenate(data['labels'], axis=0)
        data['channel_names'] = eeg_channels
        
        logger.info(f"Loaded SEED data: {data['eeg_data'].shape[0]} trials, "
                   f"{data['eeg_data'].shape[1]} channels, "
                   f"{data['eeg_data'].shape[2]} time points")
        
        return data
    
    def load_physionet_dataset(self, subject_id: Optional[int] = None) -> Dict:
        """
        Load PhysioNet cognitive workload dataset.
        
        Args:
            subject_id: Specific subject to load (None for all subjects)
            
        Returns:
            Dictionary containing EEG data, labels, and metadata
        """
        logger.info("Loading PhysioNet dataset...")
        
        physionet_dir = os.path.join(self.data_dir, 'physionet')
        if not os.path.exists(physionet_dir):
            logger.warning(f"PhysioNet directory not found: {physionet_dir}")
            return self._create_mock_data('physionet')
        
        data = {
            'eeg_data': [],
            'labels': [],
            'sampling_rate': 160,
            'channels': 64,
            'subjects': []
        }
        
        # Standard 64-channel montage
        eeg_channels = [f'Ch{i:02d}' for i in range(1, 65)]
        
        # Mock data generation
        if subject_id is None:
            subject_ids = range(1, 21)  # 20 subjects in PhysioNet
        else:
            subject_ids = [subject_id]
            
        for subj in subject_ids:
            # Generate mock EEG data for cognitive workload tasks
            mock_eeg = np.random.randn(20, 64, 12800) * 40 + np.sin(np.linspace(0, 40*np.pi, 12800)) * 20
            
            # Generate mock labels (low, medium, high cognitive load)
            mock_labels = np.random.choice([0, 1, 2], 20)  # 0: low, 1: medium, 2: high
            
            data['eeg_data'].append(mock_eeg)
            data['labels'].append(mock_labels)
            data['subjects'].append(f'physionet_subject_{subj:02d}')
        
        data['eeg_data'] = np.concatenate(data['eeg_data'], axis=0)
        data['labels'] = np.concatenate(data['labels'], axis=0)
        data['channel_names'] = eeg_channels
        
        logger.info(f"Loaded PhysioNet data: {data['eeg_data'].shape[0]} trials, "
                   f"{data['eeg_data'].shape[1]} channels, "
                   f"{data['eeg_data'].shape[2]} time points")
        
        return data
    
    def _create_mock_data(self, dataset_name: str) -> Dict:
        """
        Create mock data for demonstration purposes when real datasets are not available.
        
        Args:
            dataset_name: Name of the dataset ('deap', 'seed', 'physionet')
            
        Returns:
            Dictionary containing mock EEG data and labels
        """
        logger.info(f"Creating mock data for {dataset_name} dataset...")
        
        if dataset_name == 'deap':
            return {
                'eeg_data': np.random.randn(100, 32, 7680) * 50,
                'labels': np.random.randint(1, 10, (100, 4)),
                'sampling_rate': 128,
                'channels': 32,
                'channel_names': [f'Ch{i:02d}' for i in range(1, 33)],
                'subjects': [f'deap_mock_{i:02d}' for i in range(1, 101)]
            }
        elif dataset_name == 'seed':
            return {
                'eeg_data': np.random.randn(75, 62, 24000) * 30,
                'labels': np.random.choice([0, 1, 2], 75),
                'sampling_rate': 200,
                'channels': 62,
                'channel_names': [f'Ch{i:02d}' for i in range(1, 63)],
                'subjects': [f'seed_mock_{i:02d}' for i in range(1, 76)]
            }
        elif dataset_name == 'physionet':
            return {
                'eeg_data': np.random.randn(80, 64, 12800) * 40,
                'labels': np.random.choice([0, 1, 2], 80),
                'sampling_rate': 160,
                'channels': 64,
                'channel_names': [f'Ch{i:02d}' for i in range(1, 65)],
                'subjects': [f'physionet_mock_{i:02d}' for i in range(1, 81)]
            }
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing dataset information
        """
        dataset_info = {
            'deap': {
                'name': 'DEAP Dataset',
                'description': 'EEG-based emotion recognition',
                'subjects': 32,
                'channels': 32,
                'sampling_rate': 128,
                'trials_per_subject': 40,
                'duration': 60,  # seconds
                'labels': ['Valence', 'Arousal', 'Dominance', 'Liking']
            },
            'seed': {
                'name': 'SEED Dataset',
                'description': 'EEG-based affective state analysis',
                'subjects': 15,
                'channels': 62,
                'sampling_rate': 200,
                'trials_per_subject': 15,
                'duration': 120,  # seconds
                'labels': ['Negative', 'Neutral', 'Positive']
            },
            'physionet': {
                'name': 'PhysioNet Cognitive Workload',
                'description': 'EEG-based cognitive workload assessment',
                'subjects': 20,
                'channels': 64,
                'sampling_rate': 160,
                'trials_per_subject': 20,
                'duration': 80,  # seconds
                'labels': ['Low Load', 'Medium Load', 'High Load']
            }
        }
        
        return dataset_info.get(dataset_name, {})


def main():
    """Example usage of the EEGDataLoader."""
    loader = EEGDataLoader()
    
    # Load different datasets
    print("=== Dataset Information ===")
    for dataset in ['deap', 'seed', 'physionet']:
        info = loader.get_dataset_info(dataset)
        print(f"\n{info['name']}:")
        print(f"  Description: {info['description']}")
        print(f"  Subjects: {info['subjects']}")
        print(f"  Channels: {info['channels']}")
        print(f"  Sampling Rate: {info['sampling_rate']} Hz")
    
    # Load sample data
    print("\n=== Loading Sample Data ===")
    deap_data = loader.load_deap_dataset(subject_id=1)
    print(f"DEAP sample shape: {deap_data['eeg_data'].shape}")
    print(f"DEAP labels shape: {deap_data['labels'].shape}")


if __name__ == "__main__":
    main()
