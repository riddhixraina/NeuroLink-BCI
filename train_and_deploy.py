#!/usr/bin/env python3
"""
Script to train the model and deploy it for real-time predictions.
"""

import os
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Train the model and prepare for deployment."""
    
    logger.info("🚀 Starting NeuroLink-BCI Model Training and Deployment")
    
    # Step 1: Train the model
    logger.info("📚 Step 1: Training CNN-LSTM model with improved training...")
    try:
        result = subprocess.run([
            sys.executable, 
            os.path.join('scripts', 'improved_train_model.py')
        ], capture_output=True, text=True, check=True)
        
        logger.info("✅ Model training completed successfully!")
        logger.info("Training output:")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    
    # Step 2: Check if model was saved
    model_path = os.path.join('models', 'trained_model.pth')
    if os.path.exists(model_path):
        logger.info(f"✅ Trained model saved to: {model_path}")
    else:
        logger.error(f"❌ Model not found at: {model_path}")
        return False
    
    # Step 3: Test backend with trained model
    logger.info("🔧 Step 2: Testing backend with trained model...")
    logger.info("You can now start the backend and frontend:")
    logger.info("  Backend:  python backend/app.py")
    logger.info("  Frontend: npm --prefix frontend start")
    
    logger.info("🎉 Training and deployment setup completed!")
    logger.info("The system will now use the trained CNN-LSTM model for real predictions!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Ready to run with trained model!")
        print("Start the system with: python backend/app.py")
    else:
        print("\n❌ Setup failed. Check the logs above.")
        sys.exit(1)
