import sys
import os
from dotenv import load_dotenv
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load environment variables from .env file
load_dotenv()

import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Add src path
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src2')
sys.path.append(src_path)

# Imports
from active_gliner.llm.backends.ollama import OllamaBackend
from active_gliner.llm.backends.cerebras import CerebrasBackend
from active_gliner.llm.prompts import StandardPrompt, StructuredPrompt
from active_gliner.llm.validation import validate_ner_response
from active_gliner.llm.stats import ValidationStats
from active_gliner.llm.exceptions import HardQuotaError
from active_gliner.config.data_paths import (
    MIT_movies_NER_train_path,
    MIT_movies_NER_labels_path
)
from active_gliner.create_data.gliner_format import (
    convert_raw_json_to_gliner_training,
    convert_llm_entities_to_gliner_predictions
)
from active_gliner.evaluate_model.get_metrics import evaluate_with_ground_truth

# Configuration
BACKEND = "cerebras"  # Change to "cerebras" to use Cerebras
NUM_EXAMPLES = 10  # Number of examples to label
PROMPT_TYPE = "standard"  # "standard" or "structured"

# Create output directory
output_dir = Path("/app/data/llm_labels")
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("\n" + "="*80)
print("LOADING DATA")


with open(MIT_movies_NER_train_path, 'r') as f:
    train_data = json.load(f)

with open(MIT_movies_NER_labels_path, 'r') as f:
    labels = json.load(f)

print(f"Train examples: {len(train_data)}")
print(f"Entity types: {labels}")
print(f"Labeling first {NUM_EXAMPLES} examples with {BACKEND} backend using {PROMPT_TYPE} prompt")

# Initialize backend
print("\n" + "="*80)
print(f"INITIALIZING {BACKEND.upper()} BACKEND")


if BACKEND == "ollama":
    # Uses default model from OLLAMA_DEFAULT config (gemma3:12b)
    backend = OllamaBackend()
elif BACKEND == "cerebras":
    # Uses default model from CEREBRAS_DEFAULT config (qwen-3-235b-a22b-instruct-2507)
    backend = CerebrasBackend()
else:
    raise ValueError(f"Unknown backend: {BACKEND}")

print(f"Backend initialized: {backend.config['model_name']}")

# Generate labels
print("\n" + "="*80)
print("GENERATING LABELS")


validation_stats = ValidationStats()
results = []
llm_predictions = []
successful_labels = 0

train_subset = train_data[:NUM_EXAMPLES]
converted_train_data = convert_raw_json_to_gliner_training(train_subset)

for i, example in enumerate(train_subset):
    text = example['sentence']
    print(f"\n[{i+1}/{NUM_EXAMPLES}] Text: {text}")

    try:
        # Generate prompt
        if PROMPT_TYPE == "standard":
            prompt = StandardPrompt(text=text, entities=labels)
            content, stats = backend.generate(prompt)
        else:  # structured
            prompt, schema = StructuredPrompt(text=text, entities=labels)
            content, stats = backend.generate(prompt, schema=schema)

        print(f"llm response:\n{content}")
        print(f"  Generated in {stats['latency_ms']:.0f}ms (tokens: {stats['input_tokens']}+{stats['output_tokens']})")

        # Validate response
        valid, data, errors = validate_ner_response(
            response_text=content,
            expected_entities=labels,
            original_text=text,
            stats=validation_stats
        )

        if valid:
            successful_labels += 1
            print(f"  Valid! Extracted {len(data['entities'])} entities which are {data['entities']}")

            char_predictions = convert_llm_entities_to_gliner_predictions(data['entities'], text)
            llm_predictions.append(char_predictions)

            print(f"  Converted llm predictions to gliner predictions: {char_predictions}")

            results.append({
                'text': text,
                'entities': data['entities'],
                'stats': stats
            })
        else:
            print(f"  Invalid response: {errors}")
            llm_predictions.append([])
            results.append({
                'text': text,
                'entities': [],
                'stats': stats,
                'errors': errors
            })

    except HardQuotaError as e:
        print(f"  Hard quota exceeded: {e}")
        print(f"Stopping labeling. Processed {i+1} examples.")
        break

    except Exception as e:
        print(f"  Error: {e}")
        llm_predictions.append([])
        results.append({
            'text': text,
            'entities': [],
            'error': str(e)
        })

# Save results
model_name = backend.config['model_name'].replace(':', '_').replace('/', '_').replace('-', '_')
model_dir = output_dir / model_name
model_dir.mkdir(parents=True, exist_ok=True)
output_file = model_dir / f"labels_{NUM_EXAMPLES}.json"

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("SUMMARY")


# Backend stats
backend_summary = backend.stats.summary()
print("\nBackend Stats:")
print(f"  Total requests: {backend_summary['total_requests']}")
print(f"  Success rate: {backend_summary['success_rate']:.2%}")
print(f"  Total tokens: {backend_summary['total_tokens']}")
print(f"  Total cost: ${backend_summary['total_cost_usd']:.4f}")
print(f"  Avg latency: {backend_summary['avg_latency_ms']:.0f}ms")

# Validation stats
validation_summary = validation_stats.summary()
print("\nValidation Stats:")
print(f"  Total validated: {validation_summary['total_validated']}")
print(f"  Validation rate: {validation_summary['validation_rate']:.2%}")
print(f"  Total entities extracted: {validation_summary['total_entities_extracted']}")
print(f"  Entity validity rate: {validation_summary['entity_validity_rate']:.2%}")

print(f"\nResults saved to: {output_file}")
print(f"Successfully labeled: {successful_labels}/{NUM_EXAMPLES}")

print("\n" + "="*80)
print("EVALUATE WITH GROUND TRUTH")

llm_results = evaluate_with_ground_truth(
    predictions=llm_predictions,
    data=converted_train_data,
    entity_types=labels,
    has_confidence=False
)

# Print results
print("\nOVERALL METRICS:")
for key, value in llm_results['overall_metrics'].items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

print("\nCLASSIFICATION REPORT:")
print(llm_results['classification_report_df'].to_string(index=False))


# count ground‑truth entities in the same 10 examples
gt_entities = []
for ex in train_data[:NUM_EXAMPLES]:
    gt_entities.extend(ex.get('entities', []))
print("Ground‑truth entities in 10 examples:", len(gt_entities))


for i, pred in enumerate(llm_predictions):
    # ground‑truth labels for this example
    gt_labels = {e['label'] for e in converted_train_data[i].get('entities', [])}
    # find predictions that don't have a matching ground‑truth label
    unmatched = [p for p in pred if p['label'] not in gt_labels]
    if unmatched:
        print(f"Example {i+1} has false positives:")
        for p in unmatched:
            print(f"  -> {p['text']} ({p['label']})")


gt_count = sum(len(ex['ner']) for ex in converted_train_data)
print("Converted ground‑truth entity count:", gt_count)  # should be 15


