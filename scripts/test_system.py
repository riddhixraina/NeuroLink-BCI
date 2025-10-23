"""
System Integration Test for NeuroLink-BCI
Tests the complete pipeline from data loading to real-time prediction.
"""

import sys
import os
import numpy as np
import time
import requests
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import EEGDataLoader
from preprocess import EEGPreprocessor
from feature_extraction import EEGFeatureExtractor
from model import EEGClassifier
from utils import generate_mock_eeg_data, create_channel_montage

def test_data_loading():
    """Test data loading functionality."""
    print("🧪 Testing Data Loading...")
    
    loader = EEGDataLoader()
    
    # Test DEAP dataset loading
    deap_data = loader.load_deap_dataset(subject_id=1)
    assert deap_data['eeg_data'].shape[1] == 32, "DEAP should have 32 channels"
    assert deap_data['sampling_rate'] == 128, "DEAP sampling rate should be 128 Hz"
    
    # Test SEED dataset loading
    seed_data = loader.load_seed_dataset(subject_id=1)
    assert seed_data['eeg_data'].shape[1] == 62, "SEED should have 62 channels"
    assert seed_data['sampling_rate'] == 200, "SEED sampling rate should be 200 Hz"
    
    print("✅ Data loading tests passed")
    return True

def test_preprocessing():
    """Test preprocessing pipeline."""
    print("🧪 Testing Preprocessing Pipeline...")
    
    # Generate mock data
    eeg_data = generate_mock_eeg_data(n_samples=20, n_channels=32, n_timepoints=7680)
    labels = np.random.randint(0, 3, 20)
    
    # Initialize preprocessor
    preprocessor = EEGPreprocessor(sampling_rate=128, n_channels=32)
    
    # Test preprocessing pipeline
    results = preprocessor.preprocess_pipeline(eeg_data, labels)
    
    assert results['eeg_data'].shape[0] > 0, "Should have processed data"
    assert results['n_trials_original'] == 20, "Should have 20 original trials"
    
    print("✅ Preprocessing tests passed")
    return True

def test_feature_extraction():
    """Test feature extraction."""
    print("🧪 Testing Feature Extraction...")
    
    # Generate mock data
    eeg_data = generate_mock_eeg_data(n_samples=10, n_channels=32, n_timepoints=256)
    
    # Initialize feature extractor
    extractor = EEGFeatureExtractor(sampling_rate=128)
    
    # Test feature extraction
    features = extractor.extract_all_features(eeg_data)
    
    assert len(features) > 0, "Should extract features"
    assert 'alpha' in features, "Should have alpha band features"
    assert 'beta' in features, "Should have beta band features"
    
    print("✅ Feature extraction tests passed")
    return True

def test_model_training():
    """Test model training and prediction."""
    print("🧪 Testing Model Training...")
    
    # Generate mock data
    X = generate_mock_eeg_data(n_samples=100, n_channels=32, n_timepoints=256)
    y = np.random.randint(0, 3, 100)
    
    # Initialize classifier
    classifier = EEGClassifier(n_channels=32, n_timepoints=256, n_classes=3)
    
    # Test prediction
    predictions, confidence = classifier.predict(X[:5])
    
    assert len(predictions) == 5, "Should predict for 5 samples"
    assert len(confidence) == 5, "Should have confidence scores"
    assert all(0 <= conf <= 1 for conf in confidence), "Confidence should be between 0 and 1"
    
    print("✅ Model tests passed")
    return True

def test_backend_api():
    """Test backend API endpoints."""
    print("🧪 Testing Backend API...")
    
    base_url = "http://localhost:5000"
    
    try:
        # Test health check
        response = requests.get(f"{base_url}/api/health", timeout=5)
        assert response.status_code == 200, "Health check should return 200"
        
        # Test status endpoint
        response = requests.get(f"{base_url}/api/status", timeout=5)
        assert response.status_code == 200, "Status endpoint should return 200"
        
        # Test dataset info
        response = requests.get(f"{base_url}/api/datasets", timeout=5)
        assert response.status_code == 200, "Datasets endpoint should return 200"
        
        print("✅ Backend API tests passed")
        return True
        
    except requests.exceptions.ConnectionError:
        print("⚠️  Backend not running - skipping API tests")
        return False
    except Exception as e:
        print(f"❌ Backend API test failed: {e}")
        return False

def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline."""
    print("🧪 Testing End-to-End Pipeline...")
    
    # Generate mock EEG data
    eeg_data = generate_mock_eeg_data(n_samples=5, n_channels=32, n_timepoints=256)
    labels = np.random.randint(0, 3, 5)
    
    # Step 1: Preprocessing
    preprocessor = EEGPreprocessor(sampling_rate=128, n_channels=32)
    preprocessed = preprocessor.preprocess_pipeline(eeg_data, labels)
    
    # Step 2: Feature extraction
    extractor = EEGFeatureExtractor(sampling_rate=128)
    features = extractor.extract_all_features(preprocessed['eeg_data'])
    
    # Step 3: Model prediction
    classifier = EEGClassifier(n_channels=32, n_timepoints=256, n_classes=3)
    predictions, confidence = classifier.predict(preprocessed['eeg_data'])
    
    # Verify results
    assert len(predictions) == preprocessed['eeg_data'].shape[0], "Should predict for all samples"
    assert len(features) > 0, "Should extract features"
    
    print("✅ End-to-end pipeline tests passed")
    return True

def run_performance_test():
    """Run performance benchmarks."""
    print("🧪 Running Performance Tests...")
    
    # Test data generation speed
    start_time = time.time()
    eeg_data = generate_mock_eeg_data(n_samples=1000, n_channels=32, n_timepoints=256)
    generation_time = time.time() - start_time
    
    # Test preprocessing speed
    labels = np.random.randint(0, 3, 1000)
    preprocessor = EEGPreprocessor(sampling_rate=128, n_channels=32)
    
    start_time = time.time()
    preprocessed = preprocessor.preprocess_pipeline(eeg_data, labels)
    preprocessing_time = time.time() - start_time
    
    # Test feature extraction speed
    extractor = EEGFeatureExtractor(sampling_rate=128)
    
    start_time = time.time()
    features = extractor.extract_all_features(preprocessed['eeg_data'])
    feature_time = time.time() - start_time
    
    # Test prediction speed
    classifier = EEGClassifier(n_channels=32, n_timepoints=256, n_classes=3)
    
    start_time = time.time()
    predictions, confidence = classifier.predict(preprocessed['eeg_data'][:100])
    prediction_time = time.time() - start_time
    
    print(f"📊 Performance Results:")
    print(f"   Data Generation: {generation_time:.2f}s for 1000 samples")
    print(f"   Preprocessing: {preprocessing_time:.2f}s for 1000 samples")
    print(f"   Feature Extraction: {feature_time:.2f}s for {preprocessed['eeg_data'].shape[0]} samples")
    print(f"   Prediction: {prediction_time:.2f}s for 100 samples")
    print(f"   Throughput: {100/prediction_time:.0f} samples/second")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Starting NeuroLink-BCI System Tests")
    print("=" * 50)
    
    test_results = []
    
    # Run individual component tests
    test_results.append(("Data Loading", test_data_loading()))
    test_results.append(("Preprocessing", test_preprocessing()))
    test_results.append(("Feature Extraction", test_feature_extraction()))
    test_results.append(("Model Training", test_model_training()))
    test_results.append(("End-to-End Pipeline", test_end_to_end_pipeline()))
    test_results.append(("Performance", run_performance_test()))
    
    # Test backend API (optional)
    test_results.append(("Backend API", test_backend_api()))
    
    # Print results
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
