"""
Simple test - just test what we have built so far
"""
import sys
import os
sys.path.append('src')

print("🧪 Testing Modular Structure")
print("="*50)

# Test 1: Basic imports
print("1️⃣ Testing imports...")
try:
    from utils.device_setup import setup_device
    from preprocess.data_configuration import BATCH_SIZE, GLOBAL_SEED
    from preprocess.data_transformation import tokenize_text
    from evaluation.active_learning import get_lowest_score_examples_sorted
    from lora.lora_parameters import get_lora_config
    print("✅ All imports work!")
except Exception as e:
    print(f"❌ Import failed: {e}")

# Test 2: Device setup
print("\n2️⃣ Testing device setup...")
try:
    device = setup_device()
    print(f"✅ Device: {device}")
    print(f"📊 BATCH_SIZE: {BATCH_SIZE}")
    print(f"🎲 SEED: {GLOBAL_SEED}")
except Exception as e:
    print(f"❌ Device setup failed: {e}")

# Test 3: Tokenization
print("\n3️⃣ Testing tokenization...")
try:
    text = "Tom Hanks starred in Forrest Gump"
    tokens = tokenize_text(text)
    print(f"✅ Text: {text}")
    print(f"📝 Tokens: {tokens}")
except Exception as e:
    print(f"❌ Tokenization failed: {e}")

# Test 4: Active learning
print("\n4️⃣ Testing active learning...")
try:
    # Simple test data
    results = {
        "all_predictions": [
            {"scores": [0.9], "tokenized_text": ["high", "score"]},
            {"scores": [0.2], "tokenized_text": ["low", "score"]},
            {"scores": [0.6], "tokenized_text": ["medium", "score"]},
        ]
    }
    low_examples = get_lowest_score_examples_sorted(results, n=2)
    print(f"✅ Got {len(low_examples)} low confidence examples")
    for i, ex in enumerate(low_examples):
        score = min(ex['scores'])
        print(f"   {i+1}. Score: {score} - {' '.join(ex['tokenized_text'])}")
except Exception as e:
    print(f"❌ Active learning failed: {e}")

# Test 5: LoRA config
print("\n5️⃣ Testing LoRA config...")
try:
    config = get_lora_config()
    print(f"✅ LoRA r: {config.r}, alpha: {config.lora_alpha}")
    print(f"🎯 Target modules: {len(config.target_modules)}")
except Exception as e:
    print(f"❌ LoRA config failed: {e}")

print("\n🎉 Basic functionality test complete!")
print("💡 User can modify parameters in src/preprocess/data_configuration.py")
print("💡 User can change file paths in data loading functions")