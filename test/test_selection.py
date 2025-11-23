import sys
import os
import json
import torch
import random
from pathlib import Path
from gliner import GLiNER
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
from active_gliner.config.data_paths import (
    MIT_movies_NER_train_path,
    MIT_movies_NER_labels_path
)
from active_gliner.selection.strategy import calculate_mse_score, calculate_min_score, calculate_avg_score



# Load data
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

with open(MIT_movies_NER_train_path, 'r') as f:
    train_data = json.load(f)

with open(MIT_movies_NER_labels_path, 'r') as f:
    labels = json.load(f)


# Configuration
NUM_EXAMPLES = len(train_data)
STRATEGIES = ['mse', 'min', 'avg']
THRESHOLD = 0.5

print(f"Train examples: {len(train_data)}")
print(f"Entity types: {labels}")
print(f"Processing first {NUM_EXAMPLES} examples")

# Load base GLiNER model
print("\n" + "="*80)
print("LOADING BASE GLINER MODEL")
print("="*80)

model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
model.to(device)
model.eval()

print(f"Model loaded: knowledgator/modern-gliner-bi-large-v1.0")

# Generate predictions
print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80)

results = []
train_subset = train_data[:NUM_EXAMPLES]

for i, example in enumerate(train_subset):
    text = example['sentence']
    print(f"[{i+1}/{NUM_EXAMPLES}] Processing: {text[:80]}...")

    predictions = model.predict_entities(text, labels, threshold=THRESHOLD, flat_ner=False)

    # Calculate strategy scores
    mse_score = calculate_mse_score(predictions)
    min_score = calculate_min_score(predictions)
    avg_score = calculate_avg_score(predictions)

    # Store result with all information
    result = {
        'text': text,
        'entities': predictions,
        'mse': mse_score,
        'min': min_score,
        'avg': avg_score
    }

    results.append(result)

    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{NUM_EXAMPLES} examples")

print(f"\nCompleted predictions for {len(results)} examples")

# Create output directory structure
experiments_dir = Path("/app/data/experiment_data")
experiments_dir.mkdir(parents=True, exist_ok=True)

# Save sorted results for each strategy
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

for strategy in STRATEGIES:
    strategy_dir = experiments_dir / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)

    # Sort by strategy (ascending for min/avg, descending for mse)
    if strategy == 'mse':
        sorted_results = sorted(results, key=lambda x: x['mse'], reverse=True)
    elif strategy == 'min':
        sorted_results = sorted(results, key=lambda x: x['min'])
    elif strategy == 'avg':
        sorted_results = sorted(results, key=lambda x: x['avg'])

    output_file = strategy_dir / f"{strategy}_sorted_{NUM_EXAMPLES}_threshold_{THRESHOLD}.json"

    with open(output_file, 'w') as f:
        json.dump(sorted_results, f, indent=2)

    print(f"Saved {strategy} sorted results to: {output_file}")
    print(f"  Top uncertain example ({strategy}): {sorted_results[0][strategy]:.4f}")
    print(f"  Most certain example ({strategy}): {sorted_results[-1][strategy]:.4f}")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

total_entities = sum(len(r['entities']) for r in results)
examples_with_entities = sum(1 for r in results if len(r['entities']) > 0)

print(f"\nTotal examples: {len(results)}")
print(f"Examples with entities: {examples_with_entities}")
print(f"Total entities predicted: {total_entities}")
print(f"Average entities per example: {total_entities/len(results):.2f}")


print("\n" + "="*80)
print("DONE")
print("="*80)
