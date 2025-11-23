#!/usr/bin/env python3
"""
Confidence Analysis: GLiNER Base vs LLM on Worst Confidence Examples

Swappable experiment framework - change exp_config to swap data/model/LLM
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
import torch
import random
from pathlib import Path
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
    MIT_movies_NER_train_path,
    MIT_movies_NER_labels_path
)
from active_gliner.get_model.DefaultModel import DefaultModel
from active_gliner.evaluate_model.get_metrics import evaluate_with_ground_truth
from active_gliner.llm.stats import ValidationStats

from active_gliner.helper import (
    setup_experiment_logger,
    cleanup_memory,
    load_or_create_gliner_confidence_data,
    load_or_create_llm_labels,
    save_results,
    print_backend_stats,
    print_validation_stats
)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_on_subset(
    subset: List[Dict],
    converted_subset: List[Dict],
    predictions_lookup: Dict[str, List],
    entity_types: List[str],
    has_confidence: bool = False,
    model_name: str = "Model"
) -> Dict:
    """
    Unified evaluation using pre-loaded predictions

    Args:
        subset: List of examples to evaluate
        converted_subset: Ground truth in GLiNER format
        predictions_lookup: Dict mapping text -> predictions
        entity_types: List of entity types
        has_confidence: Whether predictions include confidence scores
        model_name: Name for logging (GLiNER/LLM)

    Returns:
        Dict with f1 and confidence metrics
    """
    predictions = []

    for i, ex in enumerate(subset):
        text = ex['sentence']

        # Get pre-loaded prediction
        if text in predictions_lookup:
            predictions.append(predictions_lookup[text])
        else:
            # Should not happen if data was loaded correctly
            logging.info(f"    WARNING: No {model_name} prediction found for: {text[:50]}...")
            predictions.append([])

        if (i + 1) % 50 == 0:
            logging.info(f"    {model_name}: {i + 1}/{len(subset)}")

    results = evaluate_with_ground_truth(
        predictions,
        converted_subset,
        entity_types,
        has_confidence=has_confidence
    )

    return {
        'f1': results['overall_metrics']['overall_f1_pct'],
        'confidence': results['overall_metrics'].get('overall_confidence_pct', 0.0)
    }


def generate_confidence_plot(results_df: pd.DataFrame, exp_config: Dict) -> Path:
    """
    Generate comparison plot for confidence experiment.

    Expected columns: no_worst_examples, gliner_f1, llm_f1, gliner_confidence

    Colors:
    - Red: LLM predictions
    - Green: GLiNER Base
    """
    # Consistent color scheme across all experiments
    LLM_COLOR = 'tab:red'
    BASE_COLOR = 'tab:green'

    data_name = exp_config['data_name']
    output_dir = exp_config['results_dir'] / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{data_name}_confidence_comparison.png"
    output_file = output_dir / filename

    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot GLiNER Base (Green)
    ax.plot(
        results_df['no_worst_examples'],
        results_df['gliner_f1'],
        marker='o',
        markersize=8,
        linewidth=3,
        label='GLiNER Base',
        color=BASE_COLOR,
        alpha=0.9
    )

    # Plot LLM (Red)
    ax.plot(
        results_df['no_worst_examples'],
        results_df['llm_f1'],
        marker='s',
        markersize=8,
        linewidth=3,
        label='LLM',
        color=LLM_COLOR,
        alpha=0.9
    )

    # Format
    ax.set_title(
        f"Confidence Analysis: GLiNER vs LLM - {data_name.upper()}\n"
        f"Performance on Worst Confidence Examples",
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Number of Worst Confidence Examples', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add value annotations at key points
    for i in [0, len(results_df)//2, -1]:
        if i < len(results_df):
            x = results_df.iloc[i]['no_worst_examples']
            y_gliner = results_df.iloc[i]['gliner_f1']
            y_llm = results_df.iloc[i]['llm_f1']

            ax.annotate(f'{y_gliner:.1f}%',
                       xy=(x, y_gliner),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9,
                       alpha=0.7,
                       color=BASE_COLOR)

            ax.annotate(f'{y_llm:.1f}%',
                       xy=(x, y_llm),
                       xytext=(5, -15),
                       textcoords='offset points',
                       fontsize=9,
                       alpha=0.7,
                       color=LLM_COLOR)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Plot saved: {output_file}")
    return output_file


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_confidence_gliner_llm_baseline_f1(exp_config: Dict):
    """Run confidence analysis experiment"""

    # Setup logging
    log_file = setup_experiment_logger(
        experiment_name=exp_config['experiment_name'],
        results_dir=exp_config['results_dir']
    )

    logging.info("="*80)
    logging.info("CONFIDENCE ANALYSIS: GLINER vs LLM")
    logging.info("="*80)
    logging.info(f"Device: {device}")
    logging.info(f"Data: {exp_config['data_name']}")
    logging.info(f"Strategy: {exp_config['strategy']}")
    logging.info(f"GLiNER: {exp_config['model_class'].__name__}")
    logging.info(f"LLM: {exp_config['llm_backend']} ({exp_config['llm_model_name']})")
    logging.info(f"Subset sizes: {exp_config['subset_sizes']}")
    logging.info(f"Log file: {log_file}")

    # Load data
    logging.info("\n" + "="*80)
    logging.info("LOADING DATA")
    logging.info("="*80)

    with open(exp_config['train_data_path'], 'r') as f:
        train_data = json.load(f)

    with open(exp_config['labels_path'], 'r') as f:
        entity_types = json.load(f)

    logging.info(f"Train data: {len(train_data)} examples")
    logging.info(f"Entity types: {entity_types}")

    # Load or generate GLiNER confidence data (will create model only if needed)
    logging.info("\n" + "="*80)
    logging.info("LOADING/GENERATING GLINER CONFIDENCE DATA")
    logging.info("="*80)

    confidence_data = load_or_create_gliner_confidence_data(
        model_class=exp_config['model_class'],
        adapter_path=exp_config['model_adapter_path'],
        train_data=train_data,
        entity_types=entity_types,
        confidence_file_path=exp_config['confidence_data_path'],
        threshold=exp_config.get('gliner_threshold', 0.5)
    )

    # Filter confidence data to keep only examples where GLiNER made predictions
    logging.info("\n" + "="*80)
    logging.info("FILTERING CONFIDENCE DATA")
    logging.info("="*80)

    train_lookup = {ex['sentence']: ex for ex in train_data}
    filtered_confidence_data = []
    skipped_no_predictions = 0

    logging.info(f"Filtering to get {exp_config['max_confidence_examples']} examples where GLiNER made predictions...")

    for conf_ex in confidence_data:
        has_gliner_pred = len(conf_ex.get('entities', [])) > 0

        if has_gliner_pred:
            # Verify ground truth exists
            text = conf_ex['text']
            if text in train_lookup:
                has_ground_truth = len(train_lookup[text].get('entities', [])) > 0
                if has_ground_truth:
                    filtered_confidence_data.append(conf_ex)
        else:
            skipped_no_predictions += 1

        # Stop when we have enough
        if len(filtered_confidence_data) >= exp_config['max_confidence_examples']:
            break

    logging.info(f"Filtered: Kept {len(filtered_confidence_data)} examples where GLiNER made predictions")
    logging.info(f"  Skipped {skipped_no_predictions} examples with no GLiNER predictions")
    strategy = exp_config['strategy']
    logging.info(f"  {strategy.upper()} range: {filtered_confidence_data[0][strategy]:.4f} to {filtered_confidence_data[-1][strategy]:.4f}")

    # Match filtered confidence examples with ground truth
    examples = []

    for conf_ex in filtered_confidence_data:
        text = conf_ex['text']
        if text in train_lookup:
            examples.append(train_lookup[text])

    converted = convert_raw_json_to_gliner_training(examples)

    logging.info(f"\nSelected {len(examples)} worst confidence examples for experiment")

    # Load or create LLM labels (will create backend only if needed)
    logging.info("\n" + "="*80)
    logging.info("LOADING/GENERATING LLM LABELS")
    logging.info("="*80)

    llm_labels, llm_validation_stats = load_or_create_llm_labels(
        examples=examples,
        max_needed=exp_config['max_confidence_examples'],
        llm_labels_dir=exp_config['llm_labels_dir'],
        entity_types=entity_types,
        backend_type=exp_config['llm_backend'],
        model_name=exp_config['llm_model_name'],
        strategy=exp_config['strategy']
    )

    # Build prediction lookups for evaluation
    logging.info("\n" + "="*80)
    logging.info("BUILDING PREDICTION LOOKUPS")
    logging.info("="*80)

    # GLiNER predictions lookup - extract from filtered confidence_data
    gliner_predictions_lookup = {
        conf['text']: conf['entities']
        for conf in filtered_confidence_data
    }
    logging.info(f"GLiNER predictions: {len(gliner_predictions_lookup)} examples")

    # LLM predictions lookup - convert entities to GLiNER format
    llm_predictions_lookup = {}
    for label in llm_labels:
        char_preds = convert_llm_entities_to_gliner_predictions(
            label['entities'],
            label['text']
        )
        llm_predictions_lookup[label['text']] = char_preds
    logging.info(f"LLM predictions: {len(llm_predictions_lookup)} examples")

    # Run experiments
    logging.info("\n" + "="*80)
    logging.info("RUNNING EXPERIMENTS")
    logging.info("="*80)

    results = {
        'no_worst_examples': [],
        'gliner_f1': [],
        'gliner_confidence': [],
        'llm_f1': []
    }

    for n in exp_config['subset_sizes']:
        logging.info(f"\n{'='*80}")
        logging.info(f"SUBSET SIZE: {n}")
        logging.info(f"{'='*80}")

        subset = examples[:n]
        converted_subset = converted[:n]

        # Evaluate GLiNER using pre-loaded predictions
        logging.info("  Evaluating GLiNER...")
        gliner_metrics = evaluate_on_subset(
            subset,
            converted_subset,
            gliner_predictions_lookup,
            entity_types,
            has_confidence=True,
            model_name="GLiNER"
        )

        # Evaluate LLM using pre-loaded predictions
        logging.info("  Evaluating LLM...")
        llm_metrics = evaluate_on_subset(
            subset,
            converted_subset,
            llm_predictions_lookup,
            entity_types,
            has_confidence=False,
            model_name="LLM"
        )

        # Store results
        results['no_worst_examples'].append(n)
        results['gliner_f1'].append(gliner_metrics['f1'])
        results['gliner_confidence'].append(gliner_metrics['confidence'])
        results['llm_f1'].append(llm_metrics['f1'])

        # Comparison
        logging.info(f"\n  Results:")
        logging.info(f"    GLiNER: F1={gliner_metrics['f1']:.2f}%, Conf={gliner_metrics['confidence']:.2f}%")
        logging.info(f"    LLM:    F1={llm_metrics['f1']:.2f}%")
        winner = 'GLiNER' if gliner_metrics['f1'] > llm_metrics['f1'] else 'LLM' if llm_metrics['f1'] > gliner_metrics['f1'] else 'TIE'
        logging.info(f"    Winner: {winner}")

        torch.cuda.empty_cache()

    # Save results
    logging.info("\n" + "="*80)
    logging.info("SAVING RESULTS")
    logging.info("="*80)

    results_df = pd.DataFrame(results)
    logging.info("\n" + results_df.to_string(index=False))

    csv_file = save_results(results_df, exp_config)
    plot_file = generate_confidence_plot(results_df, exp_config)

    logging.info(f"\nCSV:  {csv_file}")
    logging.info(f"Plot: {plot_file}")

    # Summary
    logging.info("\n" + "="*80)
    logging.info("SUMMARY")
    logging.info("="*80)
    logging.info(f"Best GLiNER F1: {max(results['gliner_f1']):.2f}%")
    logging.info(f"Best LLM F1:    {max(results['llm_f1']):.2f}%")
    logging.info(f"GLiNER predictions: {len(gliner_predictions_lookup)} examples")
    logging.info(f"LLM predictions: {len(llm_predictions_lookup)} examples")
    logging.info("\nCompleted!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    strategy = 'mse'  # Options: 'mse', 'min', 'mnlp', 'avg'

    exp_config = {
        # Data
        'data_name': 'mit_movies',
        'train_data_path': MIT_movies_NER_train_path,
        'labels_path': MIT_movies_NER_labels_path,
        'strategy': strategy,
        'confidence_data_path': f'/app/data/experiment_data/{strategy}/{strategy}_sorted_9774_threshold_0.5.json',
        'max_confidence_examples': 2500,

        # GLiNER model
        'model_class': DefaultModel,
        'model_adapter_path': None,
        'gliner_threshold': 0.5,

        # LLM
        'llm_backend': 'ollama',
        'llm_model_name': 'gemma3:12b',
        'llm_labels_dir': '/app/data/llm_labels/gemma3_12b/train_filtered',

        # Experiment
        'subset_sizes': [10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500],

        # Output
        'experiment_name': 'exp_confidence_gliner_llm_baseline_f1',
        'results_dir': Path('/app/results2/exp_confidence_gliner_llm_baseline_f1')
    }

    run_confidence_gliner_llm_baseline_f1(exp_config)
