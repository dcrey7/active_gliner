import sys
import os
import json
import torch
import random
from gliner import GLiNER
import gc

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
print(src_path)

# Device auto-detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {device}")

# Data loading
from active_gliner.create_data.gliner_format import convert_raw_json_to_gliner_training
from active_gliner.config.data_paths import (
    MIT_movies_NER_train_path,
    MIT_movies_NER_test_path,
    MIT_movies_NER_labels_path
)
from active_gliner.get_model.DefaultModel import DefaultModel


with open(MIT_movies_NER_train_path, 'r') as f:
    train_data = json.load(f)

with open(MIT_movies_NER_test_path, 'r') as f:
    test_data = json.load(f)

with open(MIT_movies_NER_labels_path, 'r') as f:
    labels = json.load(f)


# ============================================================================
# BASE MODEL
# ============================================================================

model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
model.to(device)

model.eval()
predictions=[]

for i in test_data[:5]:
    prediction=model.predict_entities((i['sentence']),labels,threshold=0.5,flat_ner=False)
    print(f"prediction is {prediction}")
    predictions.append(prediction)


# Convert test data to GLiNER format
converted_test_data = convert_raw_json_to_gliner_training(test_data)
for i in converted_test_data[:5]:
    print(i)

# Evaluate base model
results = model.evaluate(converted_test_data, entity_types=labels, batch_size=8, threshold=0.5,flat_ner=False)

# Clean up base model before training
print("\nCleaning up base model...")
del model
torch.cuda.empty_cache()
gc.collect()
print("Base model removed from GPU/memory")

# ============================================================================
# FINE-TUNING WITH DEFAULT MODEL
# ============================================================================

print("\n" + "="*80)
print("FINE-TUNING WITH DEFAULT MODEL (LoRA)")
print("="*80)


# Convert training data to GLiNER format
print(f"\nConverting {len(train_data)} training examples to GLiNER format...")
converted_train_data = convert_raw_json_to_gliner_training(train_data)
print(f"Converted {len(converted_train_data)} training examples")


# Convert test data to GLiNER format
converted_test_data = convert_raw_json_to_gliner_training(test_data)
for i in converted_test_data[:5]:
    print(i)

# Use  test data for eval during training (faster)
print(f"Using {len(converted_test_data)} examples for validation during training")

# Initialize DefaultModel
print("\nInitializing DefaultModel with LoRA...")
finetuned_model = DefaultModel()
finetuned_model.load_for_training()
print(f"Training model on device: {finetuned_model.device}")

# Set adapter save path
adapter_save_path = "/app/src2/active_gliner/models/default_model_adapter"
os.makedirs(adapter_save_path, exist_ok=True)

# Fine-tune the model
print(f"\nStarting fine-tuning on {len(converted_train_data)} examples...")
print(f"This will take several minutes...\n")

finetuned_model.fit(
    train_data=converted_train_data,
    eval_data=converted_test_data,
    adapter_save_path=adapter_save_path
)

print("\n" + "="*80)
print("FINE-TUNED MODEL EVALUATION")
print("="*80)

# After training, before evaluation
import gc
import torch

# Delete training model instance
del finetuned_model  # or whatever your training model variable is called
torch.cuda.empty_cache()
gc.collect()



# Load the fine-tuned model for inference
print("\nLoading fine-tuned model for evaluation...")
eval_model = DefaultModel()
eval_model.load_for_inference(adapter_path=adapter_save_path)
print(f"Evaluation model on device: {eval_model.device}")

# Evaluate on full test set
print(f"Evaluating on {len(converted_test_data)} test examples...")
finetuned_results = eval_model._model.evaluate(converted_test_data, entity_types=labels, batch_size=8, threshold=0.5,flat_ner=False)

print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)
print(f"\nBase Model Results:{results}")
print(f"Fine-tuned Model Results: {finetuned_results}")
print("\n" + "="*80)