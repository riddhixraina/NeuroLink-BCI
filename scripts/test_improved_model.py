#!/usr/bin/env python3
"""
Test script for the improved EEG model training.
Validates the training process and model performance.
"""

import os
import sys
import numpy as np
import torch
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


def test_model_loading():
    """Test if the trained model can be loaded correctly."""
    logger.info("Testing model loading...")
    
    model_path = os.path.join('models', 'trained_model.pth')
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if it's the improved model format
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            logger.info(f"Model config: {model_config}")
            
            # Initialize model
            model = CNNLSTMHybrid(
                model_config['n_channels'],
                model_config['n_timepoints'], 
                model_config['n_classes']
            ).to(device)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Test prediction
            test_data = torch.randn(1, model_config['n_channels'], model_config['n_timepoints']).to(device)
            with torch.no_grad():
                output = model(test_data)
                prediction = torch.argmax(output, dim=1)
                confidence = torch.nn.functional.softmax(output, dim=1).max()
            
            logger.info(f"Test prediction: {prediction.item()}, Confidence: {confidence.item():.4f}")
            
            # Check training results
            if 'test_results' in checkpoint:
                test_results = checkpoint['test_results']
                logger.info(f"Model test accuracy: {test_results['accuracy']:.4f}")
                logger.info(f"Model test F1-score: {test_results['f1_score']:.4f}")
            
            if 'cv_results' in checkpoint:
                cv_results = checkpoint['cv_results']
                logger.info(f"Cross-validation accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
            
            logger.info("✅ Model loading test passed!")
            return True
        else:
            logger.error("Model checkpoint missing required fields")
            return False
            
    except Exception as e:
        logger.error(f"Model loading test failed: {str(e)}")
        return False


def test_model_performance():
    """Test model performance on synthetic data."""
    logger.info("Testing model performance...")
    
    model_path = os.path.join('models', 'trained_model.pth')
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_config' not in checkpoint:
            logger.error("Model checkpoint missing config")
            return False
        
        model_config = checkpoint['model_config']
        model = CNNLSTMHybrid(
            model_config['n_channels'],
            model_config['n_timepoints'], 
            model_config['n_classes']
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Generate test data
        n_test_samples = 100
        test_data = torch.randn(n_test_samples, model_config['n_channels'], model_config['n_timepoints']).to(device)
        
        # Make predictions
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for i in range(n_test_samples):
                output = model(test_data[i:i+1])
                pred = torch.argmax(output, dim=1)
                conf = torch.nn.functional.softmax(output, dim=1).max()
                
                predictions.append(pred.item())
                confidences.append(conf.item())
        
        # Calculate statistics
        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        
        logger.info(f"Performance on {n_test_samples} test samples:")
        logger.info(f"Average confidence: {avg_confidence:.4f}")
        logger.info(f"Confidence std: {std_confidence:.4f}")
        logger.info(f"Min confidence: {min_confidence:.4f}")
        logger.info(f"Max confidence: {max_confidence:.4f}")
        
        # Check if confidence is reasonable
        if avg_confidence > 0.5 and min_confidence > 0.1:
            logger.info("✅ Model performance test passed!")
            return True
        else:
            logger.warning("Model confidence seems low")
            return False
            
    except Exception as e:
        logger.error(f"Model performance test failed: {str(e)}")
        return False


def test_backend_integration():
    """Test if the backend can load and use the model."""
    logger.info("Testing backend integration...")
    
    try:
        # Import backend components
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
        from app import load_trained_classifier
        
        # Load classifier
        classifier = load_trained_classifier()
        
        if classifier is None:
            logger.error("Failed to load classifier")
            return False
        
        # Test prediction
        test_data = np.random.randn(32, 256)  # 32 channels, 256 time points
        predictions, confidence = classifier.predict(test_data)
        
        logger.info(f"Backend prediction: {predictions[0]}, Confidence: {confidence[0]:.4f}")
        
        # Check if prediction is valid
        if 0 <= predictions[0] < 5 and 0 <= confidence[0] <= 1:
            logger.info("✅ Backend integration test passed!")
            return True
        else:
            logger.error("Invalid prediction or confidence")
            return False
            
    except Exception as e:
        logger.error(f"Backend integration test failed: {str(e)}")
        return False


def check_training_results():
    """Check if training results files exist and are valid."""
    logger.info("Checking training results...")
    
    results_dir = 'models'
    required_files = [
        'trained_model.pth',
        'training_results.json',
        'training_curves.png',
        'confusion_matrix.png'
    ]
    
    all_exist = True
    for file in required_files:
        file_path = os.path.join(results_dir, file)
        if os.path.exists(file_path):
            logger.info(f"✅ Found: {file}")
        else:
            logger.warning(f"❌ Missing: {file}")
            all_exist = False
    
    # Check training results JSON
    results_json_path = os.path.join(results_dir, 'training_results.json')
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, 'r') as f:
                results = json.load(f)
            
            logger.info("Training results summary:")
            logger.info(f"  Test accuracy: {results.get('test_accuracy', 'N/A'):.4f}")
            logger.info(f"  Test F1-score: {results.get('test_f1_score', 'N/A'):.4f}")
            logger.info(f"  CV accuracy: {results.get('cv_mean_accuracy', 'N/A'):.4f} ± {results.get('cv_std_accuracy', 'N/A'):.4f}")
            
        except Exception as e:
            logger.error(f"Error reading training results: {str(e)}")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    logger.info("🧪 Starting comprehensive model testing...")
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Model Performance", test_model_performance),
        ("Backend Integration", test_backend_integration),
        ("Training Results", check_training_results)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test ERROR: {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("🎯 TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Model is ready for deployment.")
        return True
    else:
        logger.error("⚠️ Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
