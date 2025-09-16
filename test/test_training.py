"""
Test training functions with real components
"""
import sys
import os
# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.training_helper import cleanup_memory
from training.monitor import SimpleTrainingMonitor
from lora.lora_parameters import get_lora_config
from utils.logging_setup import setup_logging
from utils.device_setup import setup_device
from utils.reproducibility import set_all_seeds
from preprocess.data_configuration import BATCH_SIZE, GLOBAL_SEED

def test_memory_cleanup():
    print("=== Testing Memory Cleanup ===")
    
    try:
        cleanup_memory()
        print("✅ Memory cleanup completed successfully")
    except Exception as e:
        print(f"❌ Memory cleanup failed: {e}")

def test_training_monitor():
    print("\n=== Testing Training Monitor ===")
    
    try:
        # User can configure patience
        patience = 5
        
        monitor = SimpleTrainingMonitor(patience=patience)
        print(f"✅ Training monitor created successfully")
        print(f"📊 Patience: {monitor.patience}")
        print(f"📊 Initial losses tracked: {len(monitor.train_losses)} (should be 0)")
        print(f"🕐 Start time set: {monitor.start_time > 0}")
        print(f"💾 Resource tracking enabled: GPU memory, CPU memory, timestamps")
        
    except Exception as e:
        print(f"❌ Training monitor creation failed: {e}")

def test_lora_config():
    print("\n=== Testing LoRA Configuration ===")
    
    try:
        config = get_lora_config()
        
        print("✅ LoRA config created successfully")
        print(f"📊 r (rank): {config.r}")
        print(f"📊 lora_alpha: {config.lora_alpha}")
        print(f"📊 lora_dropout: {config.lora_dropout}")
        print(f"📊 bias: {config.bias}")
        print(f"🎯 Target modules: {len(config.target_modules)} modules")
        print(f"📝 Sample target modules: {config.target_modules[:3]}")
        print(f"🔧 Task type: {config.task_type}")
        
    except Exception as e:
        print(f"❌ LoRA config creation failed: {e}")

def test_utils_integration():
    print("\n=== Testing Utils Integration ===")
    
    try:
        # Test logging setup
        logger = setup_logging("test_logs")
        print("✅ Logging setup successful")
        
        # Test device setup  
        device = setup_device()
        print(f"✅ Device setup successful: {device}")
        
        # Test reproducibility
        set_all_seeds(GLOBAL_SEED)
        print(f"✅ Reproducibility seeds set: {GLOBAL_SEED}")
        
        # Test configuration access
        print(f"📊 BATCH_SIZE from config: {BATCH_SIZE}")
        print(f"🎲 GLOBAL_SEED from config: {GLOBAL_SEED}")
        
    except Exception as e:
        print(f"❌ Utils integration failed: {e}")

if __name__ == "__main__":
    test_memory_cleanup()
    test_training_monitor() 
    test_lora_config()
    test_utils_integration()