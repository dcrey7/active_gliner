#!/usr/bin/env python3
"""
Threshold Analysis: GLiNER Base vs Fine-tuned vs LLM

Compare F1/Precision/Recall across different thresholds for GLiNER models and LLM
"""

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from gliner import GLiNER
import torch
from pathlib import Path
import random
from typing import List, Dict, Any

# Reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Add src2 to path
src_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(src_path)

device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Imports
from active_gliner.create_data.gliner_format import (
    convert_raw_json_to_gliner_training,
    convert_llm_entities_to_gliner_predictions
)
from active_gliner.config.data_paths import (
    MIT_movies_NER_test_path,
    MIT_movies_NER_labels_path
)
from active_gliner.evaluate_model.get_metrics import evaluate_with_ground_truth
from active_gliner.get_model.DefaultModel import DefaultModel
from active_gliner.llm.prompts import StandardPrompt, StructuredPrompt
from active_gliner.llm.validation import validate_ner_response
from active_gliner.llm.stats import ValidationStats
from active_gliner.llm.exceptions import HardQuotaError

from active_gliner.helper import (
    setup_experiment_logger,
    create_llm_backend,
    cleanup_memory,
    print_backend_stats,
    print_validation_stats
)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_gliner_with_thresholds(
    test_data: List[Dict],
    converted_test_data: List[Dict],
    entity_types: List[str],
    thresholds: List[float],
    model_name: str,
    adapter_path: str = None
) -> pd.DataFrame:
    """
    Evaluate GLiNER model with different thresholds.
    Creates a FRESH model for each threshold to avoid state accumulation.

    Args:
        test_data: Raw test data
        converted_test_data: Converted test data
        entity_types: List of entity types
        thresholds: List of thresholds to evaluate
        model_name: Name for logging
        adapter_path: Optional adapter path for fine-tuned model (None for base model)

    Returns: DataFrame with columns [Model, F1 Score, Precision, Recall]
    """
    results = []

    for threshold in thresholds:
        logging.info(f"\n{'='*80}")
        logging.info(f"THRESHOLD: {threshold} for {model_name}")
        logging.info(f"{'='*80}")

        # Load FRESH model for this threshold
        logging.info(f"Loading fresh model for threshold={threshold}...")
        model = DefaultModel(device=device)
        if adapter_path:
            logging.info(f"  With adapter: {adapter_path}")
            model.load_for_inference(adapter_path=adapter_path)
        else:
            logging.info(f"  Base model (no adapter)")
            model.load_for_inference()

        predictions = []
        for i, example in enumerate(test_data):
            pred = model.predict_entities(example['sentence'], entity_types, threshold=threshold, flat_ner=False)
            predictions.append(pred)

            if (i + 1) % 100 == 0:
                logging.info(f"  {i + 1}/{len(test_data)} predictions")

        # Evaluate
        eval_results = evaluate_with_ground_truth(
            predictions=predictions,
            data=converted_test_data,
            entity_types=entity_types,
            has_confidence=True
        )

        f1 = eval_results['overall_metrics']['overall_f1_pct']
        precision = eval_results['classification_report_df'].iloc[-1, 1] * 100
        recall = eval_results['classification_report_df'].iloc[-1, 2] * 100

        results.append({
            "Model": f"{model_name}, threshold={threshold}",
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall
        })

        logging.info(f"  F1: {f1:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%")

        # Cleanup model after each threshold
        logging.info(f"Cleaning up model after threshold={threshold}...")
        del model
        cleanup_memory()

    return pd.DataFrame(results)


def load_or_generate_llm_predictions(
    test_data: List[Dict],
    entity_types: List[str],
    llm_config: Dict
) -> List[List[Dict]]:
    """
    Load cached LLM predictions or generate new ones.

    Returns: List of GLiNER-format predictions
    """
    llm_cache_path = llm_config.get('llm_cache_path')

    # Try to load cached predictions
    if llm_cache_path and os.path.isfile(llm_cache_path):
        logging.info(f"\nLoading cached LLM predictions from {llm_cache_path}")
        with open(llm_cache_path, "r") as f:
            raw_predictions = json.load(f)

        predictions = []
        for ex in raw_predictions:
            pred = convert_llm_entities_to_gliner_predictions(ex['entities'], ex['text'])
            predictions.append(pred)

        logging.info(f"Loaded {len(predictions)} cached predictions")
        return predictions

    # Generate new predictions
    logging.info("\nGenerating LLM predictions...")

    backend_type = llm_config['backend']
    prompt_type = llm_config['prompt_type']
    output_dir = Path(llm_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create backend using helper function
    backend = create_llm_backend(backend_type, llm_config.get('model_name'))
    logging.info(f"Prompt type: {prompt_type}")

    validation_stats = ValidationStats()
    results = []
    predictions = []
    successful_labels = 0

    for i, example in enumerate(test_data):
        text = example['sentence']
        logging.info(f"\n[{i+1}/{len(test_data)}] Text: {text}")

        try:
            # Generate prompt
            if prompt_type == "standard":
                prompt = StandardPrompt(text=text, entities=entity_types)
                content, stats = backend.generate(prompt)
            else:
                prompt, schema = StructuredPrompt(text=text, entities=entity_types)
                content, stats = backend.generate(prompt, schema=schema)

            logging.info(f"  Generated in {stats['latency_ms']:.0f}ms (tokens: {stats['input_tokens']}+{stats['output_tokens']})")

            # Validate
            valid, data, errors = validate_ner_response(content, entity_types, text, validation_stats)

            if valid:
                successful_labels += 1
                logging.info(f"  Valid! Extracted {len(data['entities'])} entities")

                char_preds = convert_llm_entities_to_gliner_predictions(data['entities'], text)
                predictions.append(char_preds)

                results.append({
                    'text': text,
                    'entities': data['entities'],
                    'stats': stats
                })
            else:
                logging.info(f"  Invalid: {errors}")
                predictions.append([])
                results.append({
                    'text': text,
                    'entities': [],
                    'stats': stats,
                    'errors': errors
                })

        except HardQuotaError as e:
            logging.info(f"  Quota exceeded: {e}")
            logging.info(f"Stopping at {i+1} examples")
            break

        except Exception as e:
            logging.info(f"  Error: {e}")
            predictions.append([])
            results.append({
                'text': text,
                'entities': [],
                'error': str(e)
            })

    # Save results
    model_name = backend.config['model_name'].replace(':', '_').replace('/', '_').replace('-', '_')
    model_dir = output_dir / model_name / "test"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_file = model_dir / f"labels_{len(test_data)}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"\nSaved LLM results to: {output_file}")
    logging.info(f"Successfully labeled: {successful_labels}/{len(test_data)}")

    # Print stats using helper functions
    print_backend_stats(backend)
    print_validation_stats(validation_stats)

    return predictions


def evaluate_llm(
    llm_predictions: List[List[Dict]],
    converted_test_data: List[Dict],
    entity_types: List[str],
    llm_name: str
) -> pd.DataFrame:
    """
    Evaluate LLM predictions.

    Returns: DataFrame with one row [Model, F1 Score, Precision, Recall]
    """
    logging.info(f"\n{'='*80}")
    logging.info(f"EVALUATING LLM: {llm_name}")
    logging.info(f"{'='*80}")

    eval_results = evaluate_with_ground_truth(
        predictions=llm_predictions,
        data=converted_test_data,
        entity_types=entity_types,
        has_confidence=False
    )

    f1 = eval_results['overall_metrics']['overall_f1_pct']
    precision = eval_results['classification_report_df'].iloc[-1, 1] * 100
    recall = eval_results['classification_report_df'].iloc[-1, 2] * 100

    logging.info(f"  F1: {f1:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%")

    return pd.DataFrame([{
        "Model": f"LLM {llm_name}",
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    }])


# ============================================================================
# PLOTTING
# ============================================================================

def generate_comparison_plots(metrics_df: pd.DataFrame, output_dir: Path):
    """Generate F1, Precision, Recall comparison plots"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # F1 Score
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=metrics_df, x="Model", y="F1 Score")
    ax.bar_label(ax.containers[0], fmt="%.2f")
    plt.title("Model vs F1 Score - Comparison")
    plt.ylabel("F1 Score")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "model_f1_comparison.png")
    plt.close()

    # Recall
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=metrics_df, x="Model", y="Recall")
    ax.bar_label(ax.containers[0], fmt="%.2f")
    plt.title("Model vs Recall - Comparison")
    plt.ylabel("Recall")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "model_recall_comparison.png")
    plt.close()

    # Precision
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=metrics_df, x="Model", y="Precision")
    ax.bar_label(ax.containers[0], fmt="%.2f")
    plt.title("Model vs Precision - Comparison")
    plt.ylabel("Precision")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "model_precision_comparison.png")
    plt.close()


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_gliner_llm_threshold_f1(exp_config: Dict):
    """Run threshold analysis experiment"""

    # Setup logging
    log_file = setup_experiment_logger(
        experiment_name=exp_config['experiment_name'],
        results_dir=exp_config['results_dir']
    )

    logging.info("="*80)
    logging.info("THRESHOLD ANALYSIS: GLINER vs LLM")
    logging.info("="*80)
    logging.info(f"Device: {device}")
    logging.info(f"Data: {exp_config['data_name']}")
    logging.info(f"Thresholds: {exp_config['thresholds']}")
    logging.info(f"Log file: {log_file}")

    # Load data
    logging.info("\n" + "="*80)
    logging.info("LOADING DATA")
    logging.info("="*80)

    with open(exp_config['test_data_path'], 'r') as f:
        test_data = json.load(f)

    with open(exp_config['labels_path'], 'r') as f:
        entity_types = json.load(f)

    converted_test_data = convert_raw_json_to_gliner_training(test_data)

    logging.info(f"Test examples: {len(test_data)}")
    logging.info(f"Entity types: {entity_types}")

    # Initialize results
    all_metrics = []

    # Evaluate GLiNER Base
    if exp_config['evaluate_gliner_base']:
        logging.info("\n" + "="*80)
        logging.info("EVALUATING GLINER BASE MODEL")
        logging.info("="*80)

        base_metrics = evaluate_gliner_with_thresholds(
            test_data=test_data,
            converted_test_data=converted_test_data,
            entity_types=entity_types,
            thresholds=exp_config['thresholds'],
            model_name="GLiNER Base",
            adapter_path=None  # No adapter = base model
        )
        all_metrics.append(base_metrics)

    # Evaluate GLiNER Fine-tuned
    if exp_config['evaluate_gliner_ft']:
        logging.info("\n" + "="*80)
        logging.info("EVALUATING GLINER FINE-TUNED MODEL")
        logging.info("="*80)

        adapter_path = exp_config['gliner_ft_adapter_path']

        if adapter_path and os.path.exists(adapter_path):
            logging.info(f"Will use adapter: {adapter_path}")
        else:
            logging.info("Warning: No adapter found, will use base model")
            adapter_path = None

        ft_metrics = evaluate_gliner_with_thresholds(
            test_data=test_data,
            converted_test_data=converted_test_data,
            entity_types=entity_types,
            thresholds=exp_config['thresholds'],
            model_name="GLiNER FT",
            adapter_path=adapter_path
        )
        all_metrics.append(ft_metrics)

    # Evaluate LLM
    if exp_config['evaluate_llm']:
        logging.info("\n" + "="*80)
        logging.info("EVALUATING LLM")
        logging.info("="*80)

        llm_predictions = load_or_generate_llm_predictions(
            test_data=test_data,
            entity_types=entity_types,
            llm_config=exp_config['llm_config']
        )

        llm_metrics = evaluate_llm(
            llm_predictions=llm_predictions,
            converted_test_data=converted_test_data,
            entity_types=entity_types,
            llm_name=exp_config['llm_config'].get('model_name', 'LLM')
        )
        all_metrics.append(llm_metrics)

    # Combine results
    logging.info("\n" + "="*80)
    logging.info("FINAL RESULTS")
    logging.info("="*80)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    logging.info("\n" + metrics_df.to_string(index=False))

    # Save results
    csv_dir = exp_config['results_dir'] / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_file = csv_dir / f"{exp_config['data_name']}_threshold_analysis.csv"
    metrics_df.to_csv(csv_file, index=False)
    logging.info(f"\nCSV saved: {csv_file}")

    # Generate plots
    plots_dir = exp_config['results_dir'] / "plots"
    generate_comparison_plots(metrics_df, plots_dir)
    logging.info(f"Plots saved: {plots_dir}")

    logging.info("\nCompleted!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    exp_config = {
        # Data
        'data_name': 'mit_movies',
        'test_data_path': MIT_movies_NER_test_path,
        'labels_path': MIT_movies_NER_labels_path,

        # Experiment configuration
        'thresholds': [0.1, 0.5, 0.9],

        # GLiNER Base
        'evaluate_gliner_base': True,
        'gliner_base_model': 'knowledgator/modern-gliner-bi-large-v1.0',

        # GLiNER Fine-tuned
        'evaluate_gliner_ft': True,
        'gliner_ft_adapter_path': '/app/results2/exp_gliner_best_hyperparameters/adapters/trial_010',

        # LLM
        'evaluate_llm': True,
        'llm_config': {
            'backend': 'ollama',           # 'ollama' or 'cerebras'
            'model_name': 'gemma3:12b',     # Model name or None for default
            'prompt_type': 'standard',     # 'standard' or 'structured'
            'output_dir': '/app/data/llm_labels',
            # 'llm_cache_path': llm_cache_path': '/app/data/llm_labels/gemma3_4b/test/labels_2442.json',        # Path to cached predictions or None to generate
            'llm_cache_path': '/app/data/llm_labels/gemma3_12b/test/labels_2442.json',
        },

        # Output
        'experiment_name': 'exp_gliner_llm_threshold_f1',
        'results_dir': Path('/app/results2/exp_gliner_llm_threshold_f1')
    }

    run_gliner_llm_threshold_f1(exp_config)
