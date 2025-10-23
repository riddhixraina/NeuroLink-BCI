"""
Real-time EEG Data Streaming Module
Handles continuous EEG data streaming and real-time processing.
"""

import numpy as np
import threading
import time
import queue
from typing import Dict, List, Callable, Optional
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGStreamProcessor:
    """
    Real-time EEG stream processor for continuous data analysis.
    """
    
    def __init__(self, sampling_rate: int = 128, buffer_size: int = 1024):
        """
        Initialize the stream processor.
        
        Args:
            sampling_rate: EEG sampling rate in Hz
            buffer_size: Size of the processing buffer
        """
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.is_processing = False
        self.processors = []
        self.callbacks = []
        
    def add_processor(self, processor: Callable):
        """
        Add a data processor function.
        
        Args:
            processor: Function that processes EEG data chunks
        """
        self.processors.append(processor)
    
    def add_callback(self, callback: Callable):
        """
        Add a callback function for processed results.
        
        Args:
            callback: Function that receives processed results
        """
        self.callbacks.append(callback)
    
    def add_data(self, eeg_chunk: np.ndarray):
        """
        Add EEG data chunk to the processing buffer.
        
        Args:
            eeg_chunk: EEG data chunk (channels, time_points)
        """
        try:
            self.buffer.put_nowait({
                'data': eeg_chunk,
                'timestamp': datetime.now().isoformat()
            })
        except queue.Full:
            logger.warning("Buffer full, dropping oldest data")
            try:
                self.buffer.get_nowait()  # Remove oldest data
                self.buffer.put_nowait({
                    'data': eeg_chunk,
                    'timestamp': datetime.now().isoformat()
                })
            except queue.Empty:
                pass
    
    def start_processing(self):
        """Start the processing thread."""
        if self.is_processing:
            logger.warning("Processing already active")
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("EEG stream processing started")
    
    def stop_processing(self):
        """Stop the processing thread."""
        self.is_processing = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=5.0)
        
        logger.info("EEG stream processing stopped")
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_processing:
            try:
                # Get data from buffer
                if not self.buffer.empty():
                    data_packet = self.buffer.get_nowait()
                    eeg_chunk = data_packet['data']
                    timestamp = data_packet['timestamp']
                    
                    # Process data through all processors
                    results = {'timestamp': timestamp, 'data': eeg_chunk}
                    
                    for processor in self.processors:
                        try:
                            processed_result = processor(eeg_chunk)
                            results.update(processed_result)
                        except Exception as e:
                            logger.error(f"Processor error: {str(e)}")
                    
                    # Send results to all callbacks
                    for callback in self.callbacks:
                        try:
                            callback(results)
                        except Exception as e:
                            logger.error(f"Callback error: {str(e)}")
                
                else:
                    # No data available, sleep briefly
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Processing loop error: {str(e)}")
                time.sleep(0.1)


class RealTimeEEGSimulator:
    """
    Real-time EEG data simulator for testing and demonstration.
    """
    
    def __init__(self, n_channels: int = 32, sampling_rate: int = 128):
        """
        Initialize the EEG simulator.
        
        Args:
            n_channels: Number of EEG channels
            sampling_rate: Sampling rate in Hz
        """
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.is_streaming = False
        self.stream_thread = None
        self.callbacks = []
        
        # State parameters for realistic EEG simulation
        self.current_state = 0  # 0: relaxed, 1: focused, 2: stressed
        self.state_transition_prob = 0.1  # Probability of state change per second
        self.last_state_change = time.time()
        
    def add_callback(self, callback: Callable):
        """
        Add a callback function for new EEG data.
        
        Args:
            callback: Function that receives new EEG data
        """
        self.callbacks.append(callback)
    
    def start_streaming(self):
        """Start EEG data streaming."""
        if self.is_streaming:
            logger.warning("Streaming already active")
            return
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._streaming_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        
        logger.info("EEG streaming simulation started")
    
    def stop_streaming(self):
        """Stop EEG data streaming."""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=5.0)
        
        logger.info("EEG streaming simulation stopped")
    
    def set_cognitive_state(self, state: int):
        """
        Set the current cognitive state for simulation.
        
        Args:
            state: Cognitive state (0: relaxed, 1: focused, 2: stressed)
        """
        self.current_state = state
        logger.info(f"Cognitive state changed to: {state}")
    
    def _streaming_loop(self):
        """Main streaming loop."""
        chunk_size = int(self.sampling_rate * 0.5)  # 0.5 second chunks
        time_per_chunk = 0.5  # seconds
        
        while self.is_streaming:
            try:
                # Generate EEG chunk based on current cognitive state
                eeg_chunk = self._generate_eeg_chunk(chunk_size)
                
                # Create data packet
                data_packet = {
                    'timestamp': datetime.now().isoformat(),
                    'eeg_data': eeg_chunk,
                    'cognitive_state': self.current_state,
                    'sampling_rate': self.sampling_rate,
                    'chunk_size': chunk_size
                }
                
                # Send to all callbacks
                for callback in self.callbacks:
                    try:
                        callback(data_packet)
                    except Exception as e:
                        logger.error(f"Callback error: {str(e)}")
                
                # Random state transitions (reduced frequency for stability)
                if time.time() - self.last_state_change > 10.0:  # Minimum 10 seconds between changes
                    if np.random.random() < 0.1:  # 10% probability
                        self.current_state = np.random.randint(0, 5)  # Now includes all 5 states
                        self.last_state_change = time.time()
                        logger.info(f"Simulated state change to: {self.current_state}")
                
                # Sleep for real-time simulation
                time.sleep(time_per_chunk)
                
            except Exception as e:
                logger.error(f"Streaming loop error: {str(e)}")
                time.sleep(0.1)
    
    def _generate_eeg_chunk(self, chunk_size: int) -> np.ndarray:
        """
        Generate a realistic EEG chunk based on cognitive state.
        
        Args:
            chunk_size: Size of the EEG chunk
            
        Returns:
            EEG data chunk (channels, time_points)
        """
        # Generate time axis
        t = np.linspace(0, chunk_size / self.sampling_rate, chunk_size)
        
        # Initialize EEG data
        eeg_chunk = np.zeros((self.n_channels, chunk_size))
        
        # Generate different patterns based on cognitive state
        if self.current_state == 0:  # Relaxed
            # Higher alpha power, lower beta
            alpha_amplitude = 15
            beta_amplitude = 5
            theta_amplitude = 8
            gamma_amplitude = 2
        elif self.current_state == 1:  # Focused
            # Balanced alpha and beta, some gamma
            alpha_amplitude = 10
            beta_amplitude = 12
            theta_amplitude = 5
            gamma_amplitude = 3
        elif self.current_state == 2:  # Stressed
            # Higher beta and gamma, lower alpha
            alpha_amplitude = 5
            beta_amplitude = 18
            theta_amplitude = 3
            gamma_amplitude = 8
        elif self.current_state == 3:  # High Load
            # Very high beta and gamma, low alpha
            alpha_amplitude = 3
            beta_amplitude = 20
            theta_amplitude = 2
            gamma_amplitude = 12
        else:  # Low Load (state 4)
            # Low overall activity, higher alpha
            alpha_amplitude = 12
            beta_amplitude = 3
            theta_amplitude = 6
            gamma_amplitude = 1
        
        # Generate signals for each channel
        for ch in range(self.n_channels):
            # Base signal with noise
            signal = np.random.randn(chunk_size) * 2
            
            # Add frequency components based on state
            if self.current_state == 0:  # Relaxed
                signal += alpha_amplitude * np.sin(2 * np.pi * 10 * t)  # Alpha
                signal += theta_amplitude * np.sin(2 * np.pi * 6 * t)   # Theta
                signal += beta_amplitude * np.sin(2 * np.pi * 20 * t)   # Beta
            elif self.current_state == 1:  # Focused
                signal += alpha_amplitude * np.sin(2 * np.pi * 10 * t)  # Alpha
                signal += beta_amplitude * np.sin(2 * np.pi * 20 * t)   # Beta
                signal += gamma_amplitude * np.sin(2 * np.pi * 35 * t)  # Gamma
            else:  # Stressed
                signal += beta_amplitude * np.sin(2 * np.pi * 20 * t)   # Beta
                signal += gamma_amplitude * np.sin(2 * np.pi * 35 * t)  # Gamma
                signal += theta_amplitude * np.sin(2 * np.pi * 6 * t)   # Theta
            
            # Add channel-specific variations
            channel_phase = 2 * np.pi * ch / self.n_channels
            signal += 3 * np.sin(2 * np.pi * 8 * t + channel_phase)
            
            eeg_chunk[ch, :] = signal
        
        return eeg_chunk


class EEGDataBuffer:
    """
    Circular buffer for EEG data storage and retrieval.
    """
    
    def __init__(self, max_samples: int = 10000, n_channels: int = 32):
        """
        Initialize the data buffer.
        
        Args:
            max_samples: Maximum number of samples to store
            n_channels: Number of EEG channels
        """
        self.max_samples = max_samples
        self.n_channels = n_channels
        self.buffer = np.zeros((n_channels, max_samples))
        self.current_index = 0
        self.sample_count = 0
        self.lock = threading.Lock()
    
    def add_sample(self, sample: np.ndarray):
        """
        Add a new EEG sample to the buffer.
        
        Args:
            sample: EEG sample (channels, time_points)
        """
        with self.lock:
            n_timepoints = sample.shape[1]
            
            for i in range(n_timepoints):
                self.buffer[:, self.current_index] = sample[:, i]
                self.current_index = (self.current_index + 1) % self.max_samples
                self.sample_count = min(self.sample_count + 1, self.max_samples)
    
    def get_recent_data(self, n_samples: int) -> np.ndarray:
        """
        Get the most recent n_samples from the buffer.
        
        Args:
            n_samples: Number of samples to retrieve
            
        Returns:
            Recent EEG data (channels, n_samples)
        """
        with self.lock:
            if n_samples > self.sample_count:
                n_samples = self.sample_count
            
            if n_samples == 0:
                return np.zeros((self.n_channels, 0))
            
            # Calculate indices for recent data
            start_idx = (self.current_index - n_samples) % self.max_samples
            
            if start_idx + n_samples <= self.max_samples:
                # No wrap-around
                recent_data = self.buffer[:, start_idx:start_idx + n_samples]
            else:
                # Wrap-around case
                first_part = self.buffer[:, start_idx:]
                second_part = self.buffer[:, :n_samples - (self.max_samples - start_idx)]
                recent_data = np.concatenate([first_part, second_part], axis=1)
            
            return recent_data
    
    def get_all_data(self) -> np.ndarray:
        """
        Get all available data from the buffer.
        
        Returns:
            All EEG data (channels, samples)
        """
        return self.get_recent_data(self.sample_count)
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.fill(0)
            self.current_index = 0
            self.sample_count = 0


def main():
    """Example usage of the streaming modules."""
    # Create simulator
    simulator = RealTimeEEGSimulator(n_channels=32, sampling_rate=128)
    
    # Create stream processor
    processor = EEGStreamProcessor(sampling_rate=128)
    
    # Create data buffer
    buffer = EEGDataBuffer(max_samples=1000, n_channels=32)
    
    # Define processing function
    def process_eeg_data(eeg_chunk):
        """Example processing function."""
        # Calculate basic statistics
        mean_amplitude = np.mean(np.abs(eeg_chunk))
        std_amplitude = np.std(eeg_chunk)
        
        return {
            'mean_amplitude': mean_amplitude,
            'std_amplitude': std_amplitude,
            'processed_at': datetime.now().isoformat()
        }
    
    # Define callback function
    def handle_processed_data(results):
        """Example callback function."""
        print(f"Processed data: {results['timestamp']}, "
              f"Mean amplitude: {results['mean_amplitude']:.2f}")
    
    # Add processor and callback
    processor.add_processor(process_eeg_data)
    processor.add_callback(handle_processed_data)
    
    # Add buffer callback
    def buffer_callback(data_packet):
        """Callback to add data to buffer."""
        buffer.add_sample(data_packet['eeg_data'])
    
    simulator.add_callback(buffer_callback)
    
    # Start processing and streaming
    processor.start_processing()
    simulator.start_streaming()
    
    try:
        # Run for 10 seconds
        time.sleep(10)
        
        # Get recent data from buffer
        recent_data = buffer.get_recent_data(256)  # Last 2 seconds
        print(f"Buffer contains {buffer.sample_count} samples")
        print(f"Recent data shape: {recent_data.shape}")
        
    except KeyboardInterrupt:
        print("Stopping...")
    
    finally:
        # Stop streaming and processing
        simulator.stop_streaming()
        processor.stop_processing()


if __name__ == "__main__":
    main()
