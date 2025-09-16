"""
Test configuration and device setup
"""
import sys
import os
# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.device_setup import setup_device
from utils.logging_setup import setup_logging
from utils.reproducibility import set_all_seeds
from preprocess.data_configuration import BATCH_SIZE, GLOBAL_SEED

def test_device_setup():
    print("=== Testing Device Setup ===")
    
    device = setup_device()
    print(f"🖥️ Device: {device}")
    print(f"🔧 CUDA available: {device.type == 'cuda'}")

def test_configuration():
    print("\n=== Testing Configuration ===")
    
    print(f"📊 BATCH_SIZE: {BATCH_SIZE}")
    print(f"🎲 GLOBAL_SEED: {GLOBAL_SEED}")
    print("💡 User can modify these in src/preprocess/data_configuration.py")

def test_reproducibility():
    print("\n=== Testing Reproducibility ===")
    
    # User can change seed
    seed = 42
    
    set_all_seeds(seed)
    print(f"✅ All seeds set to: {seed}")

def test_logging():
    print("\n=== Testing Logging ===")
    
    logger = setup_logging("test_logs")
    logger.info("Test log message")
    print("✅ Logger created and test message logged")

if __name__ == "__main__":
    test_device_setup()
    test_configuration()
    test_reproducibility()
    test_logging()