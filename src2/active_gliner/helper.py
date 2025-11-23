import gc
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from active_gliner.create_data.gliner_format import convert_llm_entities_to_gliner_training
from active_gliner.evaluate_model.get_metrics import evaluate_with_ground_truth
from active_gliner.llm.backends.cerebras import CerebrasBackend
from active_gliner.llm.backends.ollama import OllamaBackend
from active_gliner.llm.exceptions import HardQuotaError
from active_gliner.llm.prompts import StandardPrompt
from active_gliner.llm.stats import ValidationStats
from active_gliner.llm.validation import validate_ner_response
from active_gliner.selection.strategy import (
    calculate_mse_score,
    calculate_min_score,
    calculate_avg_score,
    calculate_mnlp_score,
    STRATEGY_CONFIG
)



device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_experiment_logger(
    experiment_name: str,
    results_dir: Path,
    log_level: str = "INFO"
) -> Path:
    """
    Setup dual-output logger with timestamps for experiments.

    Input:
        experiment_name: Name of experiment
        results_dir: Root directory for results
        log_level: Logging level

    Output:
        Path to log file
    """
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{experiment_name}.log"

    # Remove all existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(file_formatter)

    # Console handler with immediate flush
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(console_formatter)
    console_handler.flush = sys.stdout.flush

    # Configure root logger
    logging.root.setLevel(getattr(logging, log_level))
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    return log_file


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


# ============================================================================
# GLINER CONFIDENCE DATA MANAGEMENT
# ============================================================================


def generate_gliner_confidence_data(
    model: Any,
    train_data: List[Dict],
    entity_types: List[str],
    threshold: float = 0.5
) -> List[Dict]:
    """
    Generate GLiNER predictions and calculate MSE scores for all training data

    Returns: List of dicts with format:
        [{'text': ..., 'entities': [...], 'mse': ...}, ...]
    """
    logging.info(f"Predicting on {len(train_data)} train examples...")
    results = []

    for i, example in enumerate(train_data):
        text = example['sentence']

        # Get GLiNER predictions
        predictions = model.predict_entities(text, entity_types, threshold=threshold, flat_ner=True)

        # Calculate MSE score
        mse_score = calculate_mse_score(predictions)

        # Store result
        result = {
            'text': text,
            'entities': predictions,
            'mse': mse_score
        }
        results.append(result)

        if (i + 1) % 100 == 0:
            logging.info(f"  Progress: {i + 1}/{len(train_data)}")

    logging.info(f"Completed predictions for {len(results)} examples")
    return results




def create_gliner_model(model_class: type, adapter_path: str = None) -> Any:
    """Create GLiNER model instance"""
    model = model_class(device=device)

    if adapter_path:
        model.load_for_inference(adapter_path=adapter_path)
        logging.info(f"Loaded with adapter: {adapter_path}")
    else:
        model.load_for_inference()
        logging.info("Loaded base model")

    return model



def load_or_create_gliner_confidence_data(
    model_class: type,
    adapter_path: str,
    train_data: List[Dict],
    entity_types: List[str],
    confidence_file_path: str,
    threshold: float = 0.5
) -> List[Dict]:
    """
    Load existing GLiNER confidence data or generate if not exists

    Only creates GLiNER model if file doesn't exist (optimization)

    Returns: List sorted by MSE (descending - worst confidence first)
    """
    confidence_file = Path(confidence_file_path)

    logging.info(f"Looking for GLiNER confidence data: {confidence_file}")

    # Check if file exists
    if confidence_file.exists():
        logging.info("Found existing file! Loading...")
        with open(confidence_file, 'r') as f:
            confidence_data = json.load(f)
        logging.info(f"Loaded {len(confidence_data)} confidence examples")
        logging.info("Skipping GLiNER model initialization (not needed)")
        return confidence_data

    else:
        logging.info("File not found. Need to generate GLiNER predictions...")
        logging.info("Initializing GLiNER model...")

        # Only create model when we need to generate predictions
        model = create_gliner_model(model_class, adapter_path)

        # Generate predictions
        results = generate_gliner_confidence_data(model, train_data, entity_types, threshold)

        # Sort by MSE (descending - worst first)
        sorted_results = sorted(results, key=lambda x: x['mse'], reverse=True)

        # Save to file
        confidence_file.parent.mkdir(parents=True, exist_ok=True)
        with open(confidence_file, 'w') as f:
            json.dump(sorted_results, f, indent=2)

        logging.info(f"Saved confidence data to: {confidence_file}")
        logging.info(f"Worst MSE: {sorted_results[0]['mse']:.4f}")
        logging.info(f"Best MSE: {sorted_results[-1]['mse']:.4f}")

        return sorted_results
    
# ============================================================================
# LLM DATA MANAGEMENT
# ============================================================================



def create_llm_backend(backend_type: str, model_name: str = None) -> Any:
    """Create LLM backend instance"""
    if backend_type == "ollama":
        backend = OllamaBackend(model_name=model_name) if model_name else OllamaBackend()
    elif backend_type == "cerebras":
        backend = CerebrasBackend(model_name=model_name) if model_name else CerebrasBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_type}")

    logging.info(f"Backend: {backend.config['model_name']}")
    return backend








def find_llm_label_files(llm_dir: Path, strategy: str = 'mse') -> List[tuple]:
    """Find all existing LLM label files for a specific strategy and extract their counts"""
    if not llm_dir.exists():
        return []

    files = []
    pattern = f"{strategy}_llm_labels_*.json"
    for file_path in llm_dir.glob(pattern):
        try:
            # Extract number from filename: {strategy}_llm_labels_100.json -> 100
            count = int(file_path.stem.split('_')[-1])
            files.append((count, file_path))
        except (ValueError, IndexError):
            continue

    return sorted(files, key=lambda x: x[0])  # Sort by count



def load_and_validate_llm_labels(file_path: Path, examples: List[Dict]) -> List[Dict]:
    """Load LLM labels and validate they match examples order"""
    with open(file_path, 'r') as f:
        labels = json.load(f)

    # Validate order matches
    for i, label_entry in enumerate(labels):
        if i >= len(examples):
            break
        if label_entry['text'] != examples[i]['sentence']:
            raise ValueError(
                f"Order mismatch at index {i}!\n"
                f"Expected: {examples[i]['sentence']}\n"
                f"Got: {label_entry['text']}\n"
                f"File may be corrupted or from different confidence_data"
            )

    logging.info(f"Loaded and validated {len(labels)} labels from {file_path.name}")
    return labels



def label_examples_with_llm(
    examples: List[Dict],
    entity_types: List[str],
    backend: Any,
    validation_stats: ValidationStats,
    start_idx: int = 0
) -> List[Dict]:
    """Label examples using LLM backend"""
    results = []

    logging.info(f"Labeling {len(examples)} examples (starting from index {start_idx})...")

    for i, ex in enumerate(examples):
        text = ex['sentence']
        global_idx = start_idx + i

        try:
            prompt = StandardPrompt(text=text, entities=entity_types)
            content, stats = backend.generate(prompt)
            valid, data, errors = validate_ner_response(content, entity_types, text, validation_stats)

            if valid:
                results.append({
                    'text': text,
                    'entities': data['entities'],
                    'stats': stats
                })
            else:
                results.append({
                    'text': text,
                    'entities': [],
                    'stats': stats,
                    'errors': errors
                })

            if (i + 1) % 10 == 0:
                logging.info(f"  Progress: {i + 1}/{len(examples)} (global index: {global_idx})")

        except HardQuotaError as e:
            logging.info(f"  Hard quota exceeded at {i + 1}: {e}")
            logging.info(f"  Stopping. Labeled {i} out of {len(examples)} examples.")
            # Pad remaining with empty results
            for j in range(i, len(examples)):
                results.append({
                    'text': examples[j]['sentence'],
                    'entities': [],
                    'error': 'Hard quota exceeded'
                })
            break

        except Exception as e:
            logging.info(f"  Error at {i + 1}: {e}")
            results.append({
                'text': text,
                'entities': [],
                'error': str(e)
            })

    return results



def save_llm_labels(labels: List[Dict], llm_dir: Path, count: int, strategy: str = 'mse'):
    """Save LLM labels to file with strategy prefix"""
    llm_dir.mkdir(parents=True, exist_ok=True)
    output_file = llm_dir / f"{strategy}_llm_labels_{count}.json"

    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=2)

    logging.info(f"Saved {count} labels to {output_file}")


def load_or_create_llm_labels(
    examples: List[Dict],
    max_needed: int,
    llm_labels_dir: str,
    entity_types: List[str],
    backend_type: str,
    model_name: str,
    strategy: str = 'mse'
) -> tuple[List[Dict], ValidationStats]:
    """
    Smart loading: find largest existing file, extend if needed

    Only creates LLM backend if labeling is needed (optimization)

    Returns:
        tuple of (list of label dicts in same order as examples, ValidationStats)
    """
    llm_dir = Path(llm_labels_dir)

    logging.info(f"Looking for LLM labels in: {llm_dir}")
    logging.info(f"Strategy: {strategy}")

    # Find existing files for this strategy
    existing_files = find_llm_label_files(llm_dir, strategy)

    if existing_files:
        logging.info(f"Found {len(existing_files)} existing label files: {[f'{count} labels' for count, _ in existing_files]}")
    else:
        logging.info("No existing label files found")

    # Find best match
    best_match = None
    for count, file_path in existing_files:
        if count == max_needed:
            # Exact match
            best_match = ('exact', count, file_path)
            break
        elif count < max_needed:
            # Keep track of largest file smaller than needed
            best_match = ('partial', count, file_path)

    # Handle different cases
    if best_match and best_match[0] == 'exact':
        # Exact match - just load, no backend needed
        logging.info(f"Found exact match: {best_match[2].name}")
        logging.info("Skipping LLM backend initialization (not needed)")
        labels = load_and_validate_llm_labels(best_match[2], examples)
        return labels, ValidationStats()  # Return empty stats since no labeling done

    elif best_match and best_match[0] == 'partial':
        # Partial match - need backend to label remaining
        count, file_path = best_match[1], best_match[2]
        logging.info(f"Found partial match: {file_path.name} ({count} labels)")
        logging.info(f"Need {max_needed - count} more labels")
        logging.info("Initializing LLM backend...")

        backend = create_llm_backend(backend_type, model_name)
        validation_stats = ValidationStats()

        existing = load_and_validate_llm_labels(file_path, examples)
        new_labels = label_examples_with_llm(
            examples[count:max_needed],
            entity_types,
            backend,
            validation_stats,
            start_idx=count
        )

        combined = existing + new_labels
        save_llm_labels(combined, llm_dir, max_needed, strategy)
        return combined, validation_stats

    else:
        # No match - need backend to label from scratch
        logging.info(f"No suitable file found. Labeling all {max_needed} examples from scratch...")
        logging.info("Initializing LLM backend...")

        backend = create_llm_backend(backend_type, model_name)
        validation_stats = ValidationStats()

        new_labels = label_examples_with_llm(
            examples[:max_needed],
            entity_types,
            backend,
            validation_stats,
            start_idx=0
        )
        save_llm_labels(new_labels, llm_dir, max_needed, strategy)
        return new_labels, validation_stats


def print_backend_stats(backend: Any):
    """
    Print backend statistics summary.

    Input:
        backend: LLM backend instance with stats attribute

    Output:
        Logs backend statistics to console/file
    """
    backend_summary = backend.stats.summary()
    logging.info("\nBackend Stats:")
    logging.info(f"  Total requests: {backend_summary['total_requests']}")
    logging.info(f"  Success rate: {backend_summary['success_rate']:.2%}")
    logging.info(f"  Total tokens: {backend_summary['total_tokens']}")
    logging.info(f"  Total cost: ${backend_summary['total_cost_usd']:.4f}")


def print_validation_stats(validation_stats: ValidationStats):
    """
    Print validation statistics summary.

    Input:
        validation_stats: ValidationStats instance

    Output:
        Logs validation statistics to console/file
    """
    validation_summary = validation_stats.summary()
    logging.info("\nValidation Stats:")
    logging.info(f"  Total validated: {validation_summary['total_validated']}")
    logging.info(f"  Validation rate: {validation_summary['validation_rate']:.2%}")
    logging.info(f"  Total entities: {validation_summary['total_entities_extracted']}")


# ============================================================================
# TRAINING
# ============================================================================

def prepare_llm_training_data(llm_labels: List[Dict]) -> List[Dict]:
    """
    Convert LLM labels to GLiNER training format

    Input: [{'text': '...', 'entities': [{'entity': '...', 'types': [...]}], ...}, ...]
    Output: [{'tokenized_text': [...], 'ner': [...], 'text': '...'}, ...]
    """
    training_data = []

    for label in llm_labels:
        converted = convert_llm_entities_to_gliner_training(
            entities=label['entities'],
            text=label['text']
        )
        training_data.append(converted)

    return training_data



def train_and_save_adapter(
    model_class: type,
    train_data: List[Dict],
    eval_data: List[Dict],
    lora_config: Dict,
    training_config: Dict,
    adapter_name: str,
    results_dir: Path
) -> Path:
    """
    Train model with LoRA and save adapter using clean BluePrint API

    Input:
        model_class: BluePrint subclass (e.g., DefaultModel)
        train_data: GLiNER format training data
        eval_data: GLiNER format eval data
        lora_config: LoRA configuration dict (overrides model default)
        training_config: Training hyperparameters (overrides model default)
        adapter_name: Name for saved adapter
        results_dir: Base directory for results

    Output:
        Path to saved adapter
    """
    # Create model instance
    model = model_class(device=device)

    # Load for training with custom LoRA config
    model.load_for_training(lora_config=lora_config)

    # Train with custom training config
    adapter_path = results_dir / "adapters" / adapter_name
    adapter_path.parent.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_data=train_data,
        eval_data=eval_data,
        adapter_save_path=str(adapter_path),
        training_config=training_config
    )

    # Cleanup checkpoints
    checkpoint_dir = adapter_path.parent / "checkpoints"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        logging.info("Cleaned up checkpoints")

    # Cleanup model from memory
    del model
    cleanup_memory()

    logging.info(f"Adapter saved to: {adapter_path}")
    return adapter_path


def evaluate_on_test_set(
    model_class: type,
    adapter_path: Path,
    test_data: List[Dict],
    converted_test: List[Dict],
    entity_types: List[str],
    threshold: float = 0.5
) -> float:
    """
    Evaluate model on test set using clean BluePrint API

    Input:
        model_class: BluePrint subclass (e.g., DefaultModel)
        adapter_path: Path to trained adapter
        test_data: Raw test data
        converted_test: GLiNER format test data
        entity_types: Entity types to predict
        threshold: Confidence threshold for predictions

    Output:
        F1 score percentage
    """
    # Load model with adapter
    model = model_class(device=device)
    model.load_for_inference(adapter_path=str(adapter_path))

    # Predict on all test examples
    predictions = []
    for i, ex in enumerate(test_data):
        preds = model.predict_entities(
            ex['sentence'],
            entity_types,
            threshold=threshold,
            flat_ner=False
        )
        predictions.append(preds)

        if (i + 1) % 100 == 0:
            logging.info(f"  Progress: {i + 1}/{len(test_data)}")

    # Evaluate
    results = evaluate_with_ground_truth(
        predictions,
        converted_test,
        entity_types,
        has_confidence=True
    )

    f1 = results['overall_metrics']['overall_f1_pct']

    # Cleanup
    del model
    cleanup_memory()

    return f1


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(results_df: pd.DataFrame, exp_config: Dict) -> Path:
    """Save results to CSV"""
    data_name = exp_config['data_name']
    model_name = exp_config['model_class'].__name__
    llm_name = exp_config['llm_model_name'].replace(':', '_')
    filename = f"{data_name}_FT_{model_name}_{llm_name}.csv"

    output_dir = exp_config['results_dir'] / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    results_df.to_csv(output_file, index=False)
    return output_file


def generate_plot(results_df: pd.DataFrame, exp_config: Dict) -> Path:
    """
    Generate comparison plot for mixed fine-tuning experiment.

    Colors: Gradient from Red (100% LLM) to Green (100% GT)
    """
    import matplotlib.colors as mcolors

    data_name = exp_config['data_name']
    model_name = exp_config['model_class'].__name__
    llm_name = exp_config['llm_model_name'].replace(':', '_')
    filename = f"{data_name}_FT_{model_name}_{llm_name}.png"

    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    pivot = results_df.pivot_table(
        index='n_actual',
        columns='mixing_ratio_pct',
        values='mixed_ft_f1',
        aggfunc='max'
    ).sort_index(axis=1)

    # Create color gradient from Red (0% GT = 100% LLM) to Green (100% GT = 0% LLM)
    LLM_COLOR = mcolors.to_rgb('tab:red')
    GT_COLOR = mcolors.to_rgb('tab:green')

    # Store colors for each ratio for later use in annotations
    ratio_colors = {}

    for ratio, series in pivot.items():
        # ratio is % GT, so interpolate from red (0% GT) to green (100% GT)
        ratio_normalized = ratio / 100.0
        color = tuple(
            LLM_COLOR[i] * (1 - ratio_normalized) + GT_COLOR[i] * ratio_normalized
            for i in range(3)
        )
        ratio_colors[ratio] = color

        ax.plot(
            series.index,
            series.values,
            marker='o',
            markersize=8,
            linewidth=3,
            label=f"{ratio}% GT / {100 - ratio}% LLM",
            color=color,
            alpha=0.9
        )

    # Format
    ax.set_title(
        f"Mixed Fine-tuning Comparison: {data_name.upper()}\n(Evaluated on Test Set)",
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Actual Training Examples (Worst Confidence)', fontsize=14)
    ax.set_ylabel('F1 Score on Test Set (%)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')

    # Add value annotations at key points for each mixing ratio
    annotation_indices = [0, len(pivot) // 2, -1]  # First, middle, last
    
    for ratio in pivot.columns:
        series = pivot[ratio]
        color = ratio_colors[ratio]
        
        for idx_pos in annotation_indices:
            if idx_pos < len(series):
                x = series.index[idx_pos]
                y = series.iloc[idx_pos]
                
                # Skip NaN values
                if pd.notna(y):
                    # Alternate text position to avoid overlap
                    # Use ratio to create vertical offset pattern
                    vertical_offset = 5 + (ratio % 20) - 10  # Varies between -5 and 15
                    
                    ax.annotate(
                        f'{y:.1f}%',
                        xy=(x, y),
                        xytext=(5, vertical_offset),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7,
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor=color, alpha=0.7, linewidth=0.5)
                    )

    plt.tight_layout()

    output_dir = exp_config['results_dir'] / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


# ============================================================================
# ACTIVE LEARNING STRATEGY DATA MANAGEMENT
# ============================================================================

import random as random_module

def generate_all_strategy_scores(
    model: Any,
    train_data: List[Dict],
    entity_types: List[str],
    threshold: float = 0.5
) -> List[Dict]:
    """
    Generate GLiNER predictions and calculate ALL strategy scores.

    Input:
        model: GLiNER model instance
        train_data: List of training examples with 'sentence' field
        entity_types: Entity types to predict
        threshold: Confidence threshold

    Output:
        List of dicts with format:
        [{'text': ..., 'entities': [...], 'mse': ..., 'min': ..., 'avg': ..., 'mnlp': ...}, ...]
    """
    logging.info(f"Predicting on {len(train_data)} train examples...")
    results = []

    for i, example in enumerate(train_data):
        text = example['sentence']
        predictions = model.predict_entities(text, entity_types, threshold=threshold, flat_ner=True)

        result = {
            'text': text,
            'entities': predictions,
            'mse': calculate_mse_score(predictions),
            'min': calculate_min_score(predictions),
            'avg': calculate_avg_score(predictions),
            'mnlp': calculate_mnlp_score(predictions)
        }
        results.append(result)

        if (i + 1) % 100 == 0:
            logging.info(f"  Progress: {i + 1}/{len(train_data)}")

    logging.info(f"Completed predictions for {len(results)} examples")
    return results


def load_or_create_strategy_data(
    model_class: type,
    adapter_path: str,
    train_data: List[Dict],
    entity_types: List[str],
    base_confidence_path: str,
    threshold: float = 0.5
) -> Dict[str, List[Dict]]:
    """
    Load or create sorted data for ALL strategies (mse, min, avg, mnlp, random).

    Input:
        model_class: BluePrint subclass
        adapter_path: Path to adapter (or None for base model)
        train_data: Raw training data
        entity_types: Entity types
        base_confidence_path: Base path for confidence data (e.g., /app/data/experiment_data)
        threshold: Confidence threshold

    Output:
        Dict mapping strategy name to sorted list of examples:
        {'mse': [...], 'min': [...], 'avg': [...], 'mnlp': [...], 'random': [...]}
    """
    base_path = Path(base_confidence_path)
    n_examples = len(train_data)

    # Check if base MSE file exists (we use this as indicator that data was generated)
    mse_file = base_path / 'mse' / f'mse_sorted_{n_examples}_threshold_{threshold}.json'

    if mse_file.exists():
        logging.info(f"Found existing confidence data at {base_path}")
        logging.info("Loading strategy files...")

        # Load MSE first
        with open(mse_file, 'r') as f:
            mse_data = json.load(f)
        logging.info(f"  Loaded mse: {len(mse_data)} examples")

        # Check if MNLP scores exist in data
        has_mnlp = 'mnlp' in mse_data[0] if mse_data else False

        if not has_mnlp:
            logging.info("  MNLP scores not found in existing data. Adding MNLP scores...")
            for ex in mse_data:
                ex['mnlp'] = calculate_mnlp_score(ex.get('entities', []))

            # Re-save MSE file with MNLP
            with open(mse_file, 'w') as f:
                json.dump(mse_data, f, indent=2)
            logging.info("  Updated MSE file with MNLP scores")

        strategy_data = {'mse': mse_data}

        # Load or create other strategies
        for strategy in ['min', 'avg']:
            strategy_file = base_path / strategy / f'{strategy}_sorted_{n_examples}_threshold_{threshold}.json'
            if strategy_file.exists():
                with open(strategy_file, 'r') as f:
                    data = json.load(f)

                # Add MNLP if missing
                if data and 'mnlp' not in data[0]:
                    for ex in data:
                        ex['mnlp'] = calculate_mnlp_score(ex.get('entities', []))
                    with open(strategy_file, 'w') as f:
                        json.dump(data, f, indent=2)

                strategy_data[strategy] = data
                logging.info(f"  Loaded {strategy}: {len(data)} examples")
            else:
                logging.info(f"  {strategy} file not found, creating from MSE data...")
                _, reverse = STRATEGY_CONFIG[strategy]
                sorted_data = sorted(mse_data, key=lambda x: x[strategy], reverse=reverse)

                strategy_dir = base_path / strategy
                strategy_dir.mkdir(parents=True, exist_ok=True)
                with open(strategy_file, 'w') as f:
                    json.dump(sorted_data, f, indent=2)

                strategy_data[strategy] = sorted_data
                logging.info(f"  Created {strategy}: {len(sorted_data)} examples")

        # Create MNLP sorted data
        mnlp_file = base_path / 'mnlp' / f'mnlp_sorted_{n_examples}_threshold_{threshold}.json'
        if mnlp_file.exists():
            with open(mnlp_file, 'r') as f:
                strategy_data['mnlp'] = json.load(f)
            logging.info(f"  Loaded mnlp: {len(strategy_data['mnlp'])} examples")
        else:
            logging.info("  Creating MNLP sorted data...")
            _, reverse = STRATEGY_CONFIG['mnlp']
            # Filter out inf values for sorting, put them at the front (most uncertain)
            inf_examples = [ex for ex in mse_data if ex['mnlp'] == float('inf')]
            finite_examples = [ex for ex in mse_data if ex['mnlp'] != float('inf')]
            sorted_finite = sorted(finite_examples, key=lambda x: x['mnlp'], reverse=reverse)
            sorted_data = inf_examples + sorted_finite

            mnlp_dir = base_path / 'mnlp'
            mnlp_dir.mkdir(parents=True, exist_ok=True)
            with open(mnlp_file, 'w') as f:
                json.dump(sorted_data, f, indent=2)

            strategy_data['mnlp'] = sorted_data
            logging.info(f"  Created mnlp: {len(sorted_data)} examples")

        # Load or create random
        random_file = base_path / 'random' / f'random_sorted_{n_examples}_threshold_{threshold}.json'
        if random_file.exists():
            with open(random_file, 'r') as f:
                data = json.load(f)
            # Add MNLP if missing
            if data and 'mnlp' not in data[0]:
                for ex in data:
                    ex['mnlp'] = calculate_mnlp_score(ex.get('entities', []))
                with open(random_file, 'w') as f:
                    json.dump(data, f, indent=2)
            strategy_data['random'] = data
            logging.info(f"  Loaded random: {len(data)} examples")
        else:
            strategy_data['random'] = create_random_sorted_data(mse_data, base_path, n_examples, threshold)

        return strategy_data

    # Need to generate data from scratch
    logging.info("Confidence data not found. Generating predictions...")
    logging.info("Initializing GLiNER model...")

    model = create_gliner_model(model_class, adapter_path)
    results = generate_all_strategy_scores(model, train_data, entity_types, threshold)

    # Cleanup model
    del model
    cleanup_memory()

    # Sort and save for each strategy
    strategy_data = {}

    for strategy, (score_fn, reverse) in STRATEGY_CONFIG.items():
        strategy_dir = base_path / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)

        sorted_results = sorted(results, key=lambda x: x[strategy], reverse=reverse)
        strategy_data[strategy] = sorted_results

        output_file = strategy_dir / f'{strategy}_sorted_{len(train_data)}_threshold_{threshold}.json'
        with open(output_file, 'w') as f:
            json.dump(sorted_results, f, indent=2)

        logging.info(f"Saved {strategy}: {output_file}")
        logging.info(f"  Most uncertain: {sorted_results[0][strategy]:.4f}")
        logging.info(f"  Most certain: {sorted_results[-1][strategy]:.4f}")

    # Create random sorted data
    strategy_data['random'] = create_random_sorted_data(results, base_path, len(train_data), threshold)

    return strategy_data


def create_random_sorted_data(
    data: List[Dict],
    base_path: Path,
    n_examples: int,
    threshold: float,
    seed: int = 42
) -> List[Dict]:
    """
    Create randomly shuffled data for random baseline.

    Input:
        data: List of examples with scores
        base_path: Base path for saving
        n_examples: Number of examples
        threshold: Threshold used
        seed: Random seed for reproducibility

    Output:
        Randomly shuffled list of examples
    """
    random_dir = base_path / 'random'
    random_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle with fixed seed
    shuffled = data.copy()
    random_module.seed(seed)
    random_module.shuffle(shuffled)

    output_file = random_dir / f'random_sorted_{n_examples}_threshold_{threshold}.json'
    with open(output_file, 'w') as f:
        json.dump(shuffled, f, indent=2)

    logging.info(f"Saved random: {output_file}")
    return shuffled


def filter_examples_with_predictions(
    strategy_data: Dict[str, List[Dict]],
    train_data: List[Dict],
    max_examples: int
) -> Dict[str, List[Dict]]:
    """
    Filter to keep only examples where GLiNER made predictions AND ground truth exists.

    Input:
        strategy_data: Dict of strategy -> sorted examples
        train_data: Original training data with ground truth
        max_examples: Maximum examples to keep per strategy

    Output:
        Dict of strategy -> filtered examples (matched with ground truth)
    """
    train_lookup = {ex['sentence']: ex for ex in train_data}

    filtered_data = {}

    for strategy, sorted_examples in strategy_data.items():
        filtered = []
        skipped_no_pred = 0
        skipped_no_gt = 0

        for conf_ex in sorted_examples:
            # Check if GLiNER made predictions
            if len(conf_ex.get('entities', [])) == 0:
                skipped_no_pred += 1
                continue

            # Check if ground truth exists
            text = conf_ex['text']
            if text not in train_lookup:
                skipped_no_gt += 1
                continue

            gt_ex = train_lookup[text]
            if len(gt_ex.get('entities', [])) == 0:
                skipped_no_gt += 1
                continue

            # Keep this example with ground truth attached
            filtered.append({
                'confidence_data': conf_ex,
                'ground_truth': gt_ex
            })

            if len(filtered) >= max_examples:
                break

        filtered_data[strategy] = filtered
        logging.info(f"  {strategy}: {len(filtered)} examples (skipped {skipped_no_pred} no pred, {skipped_no_gt} no GT)")

    return filtered_data


def generate_active_learning_plot(
    results_df: pd.DataFrame,
    exp_config: Dict
) -> Path:
    """
    Generate learning curves plot for active learning strategies.

    Input:
        results_df: DataFrame with columns [subset_size, strategy, f1]
        exp_config: Experiment configuration

    Output:
        Path to saved plot
    """
    import matplotlib.colors as mcolors

    data_name = exp_config['data_name']
    filename = f"{data_name}_active_learning_strategies.png"

    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Color scheme for strategies
    STRATEGY_COLORS = {
        'mse': 'tab:blue',
        'min': 'tab:orange',
        'mnlp': 'tab:purple',
        'random': 'tab:gray'
    }

    STRATEGY_LABELS = {
        'mse': 'MSE (Mean Squared Error)',
        'min': 'Least Confidence (Min)',
        'mnlp': 'MNLP (Max Norm Log Prob)',
        'random': 'Random Baseline'
    }

    # Plot each strategy
    for strategy in exp_config['strategies']:
        strategy_df = results_df[results_df['strategy'] == strategy]
        if strategy_df.empty:
            continue

        color = STRATEGY_COLORS.get(strategy, 'tab:gray')
        label = STRATEGY_LABELS.get(strategy, strategy)

        ax.plot(
            strategy_df['subset_size'],
            strategy_df['f1'],
            marker='o',
            markersize=8,
            linewidth=3,
            label=label,
            color=color,
            alpha=0.9
        )

    # Format
    ax.set_title(
        f"Active Learning Strategy Comparison: {data_name.upper()}\n"
        f"(Fine-tuned on Ground Truth Labels, Evaluated on Test Set)",
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Number of Training Examples', fontsize=14)
    ax.set_ylabel('F1 Score on Test Set (%)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='lower right')

    # Add annotations at key points
    for strategy in exp_config['strategies']:
        strategy_df = results_df[results_df['strategy'] == strategy]
        if strategy_df.empty:
            continue

        color = STRATEGY_COLORS.get(strategy, 'tab:gray')

        # Annotate first and last points
        for idx in [0, -1]:
            if abs(idx) < len(strategy_df):
                row = strategy_df.iloc[idx]
                x, y = row['subset_size'], row['f1']

                ax.annotate(
                    f'{y:.1f}%',
                    xy=(x, y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8,
                    color=color,
                    fontweight='bold'
                )

    plt.tight_layout()

    output_dir = exp_config['results_dir'] / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file