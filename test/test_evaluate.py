import sys
import os
import json
import torch
import random
import warnings
warnings.filterwarnings('ignore')

# Set seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add src path
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src2')
sys.path.append(src_path)

# Device
device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Imports
import active_gliner as ag
from active_gliner.create_data.gliner_format import convert_raw_json_to_gliner_training
from active_gliner.config.data_paths import (
    MIT_movies_NER_train_path,
    MIT_movies_NER_test_path,
    MIT_movies_NER_labels_path
)
from active_gliner.get_model.DefaultModel import DefaultModel
from active_gliner.evaluate_model.get_metrics import (
    evaluate_with_ground_truth,
    evaluate_without_ground_truth
)

# Load data
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

with open(MIT_movies_NER_train_path, 'r') as f:
    train_data = json.load(f)

with open(MIT_movies_NER_test_path, 'r') as f:
    test_data = json.load(f)

with open(MIT_movies_NER_labels_path, 'r') as f:
    labels = json.load(f)

print(f"Train examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")
print(f"Entity types: {labels}")

# Convert to GLiNER format
converted_test_data = convert_raw_json_to_gliner_training(test_data)

# Load model
adapter_path = "/app/src2/active_gliner/models/default_model_adapter"
model = DefaultModel(device=device)

if os.path.exists(adapter_path):
    print(f"\nLoading fine-tuned model from {adapter_path}")
    model.load_for_inference(adapter_path=adapter_path)
else:
    print("\nLoading base model (no adapter found)")
    model.load_for_inference()

# ============================================================================
# TEST 1: EVALUATE WITH GROUND TRUTH (Test Set)
# ============================================================================

print("\n" + "="*80)
print("TEST 1: EVALUATE WITH GROUND TRUTH (Test Set)")
print("="*80)

# Get predictions on test set
test_texts = [example['sentence'] for example in test_data]
print(f"Generating predictions for {len(test_texts)} examples...")
test_predictions = []
for text in test_texts:
    # Use flat_ner=False to match GLiNER's evaluate() behavior
    pred = model.predict_entities(text, labels, threshold=0.5, flat_ner=False)
    test_predictions.append(pred)

# Evaluate
test_results = evaluate_with_ground_truth(
    predictions=test_predictions,
    data=converted_test_data,
    entity_types=labels,
    has_confidence=True
)

# Print results
print("\nOVERALL METRICS:")
for key, value in test_results['overall_metrics'].items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

print("\nCLASSIFICATION REPORT:")
print(test_results['classification_report_df'].to_string(index=False))

finetuned_results = model._model.evaluate(converted_test_data, entity_types=labels, flat_ner=False, batch_size=8, threshold=0.5)
print(f"\nFine-tuned Model Results (flat_ner=False): {finetuned_results}")

# ============================================================================
# TEST 2: EVALUATE WITHOUT GROUND TRUTH (2000 Train Samples)
# ============================================================================

print("\n" + "="*80)
print("TEST 2: EVALUATE WITHOUT GROUND TRUTH (2000 Train Samples)")
print("="*80)

# Use 2000 train samples
train_subset = train_data[:2000]
train_texts = [example['sentence'] for example in train_subset]

# Get predictions
print(f"Generating predictions for {len(train_texts)} examples...")
train_predictions = []
for text in train_texts:
    # Use flat_ner=False to match GLiNER's evaluate() behavior
    pred = model.predict_entities(text, labels, threshold=0.5, flat_ner=False)
    train_predictions.append(pred)

# Evaluate
train_results = evaluate_without_ground_truth(
    predictions=train_predictions,
    entity_types=labels
)

# Print results
print("\nOVERALL METRICS:")
for key, value in train_results['overall_metrics'].items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

print(f"\nHigh Confidence Examples: {len(train_results['high_confidence_examples'])}")
print(f"Low Confidence Examples: {len(train_results['low_confidence_examples'])}")

print("\n" + "="*80)
print("DONE")
print("="*80)
