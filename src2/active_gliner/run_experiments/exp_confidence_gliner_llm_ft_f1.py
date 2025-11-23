#!/usr/bin/env python3
"""
Fine-tuning Comparison: GLiNER trained on LLM labels vs Ground Truth labels

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
from typing import Dict

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
from active_gliner.create_data.gliner_format import convert_raw_json_to_gliner_training
from active_gliner.config.data_paths import (
    MIT_movies_NER_train_path,
    MIT_movies_NER_test_path,
    MIT_movies_NER_labels_path
)
from active_gliner.config.model_configs import DEFAULT_LORA_CONFIG, DEFAULT_TRAINING_CONFIG
from active_gliner.get_model.DefaultModel import DefaultModel

from active_gliner.helper import (
    setup_experiment_logger,
    cleanup_memory,
    load_or_create_gliner_confidence_data,
    load_or_create_llm_labels,
    prepare_llm_training_data,
    train_and_save_adapter,
    evaluate_on_test_set,
    save_results
)


# ============================================================================
# PLOTTING
# ============================================================================

def generate_ft_comparison_plot(results_df: pd.DataFrame, exp_config: Dict) -> Path:
    """
    Generate comparison plot for FT experiment.

    Expected columns: n_requested, gliner_llm_ft_f1, gliner_gt_ft_f1

    Colors:
    - Red: GLiNER trained on LLM labels
    - Green: GLiNER trained on Ground Truth labels
    """
    # Consistent color scheme across all experiments
    LLM_COLOR = 'tab:red'
    GT_COLOR = 'tab:green'

    data_name = exp_config['data_name']
    output_dir = exp_config['results_dir'] / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{data_name}_ft_comparison.png"
    output_file = output_dir / filename

    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot GLiNER FT with LLM labels (Blue)
    ax.plot(
        results_df['n_requested'],
        results_df['gliner_llm_ft_f1'],
        marker='o',
        markersize=8,
        linewidth=3,
        label='GLiNER FT (LLM Labels)',
        color=LLM_COLOR,
        alpha=0.9
    )

    # Plot GLiNER FT with Ground Truth (Orange)
    ax.plot(
        results_df['n_requested'],
        results_df['gliner_gt_ft_f1'],
        marker='s',
        markersize=8,
        linewidth=3,
        label='GLiNER FT (Ground Truth)',
        color=GT_COLOR,
        alpha=0.9
    )

    # Format
    ax.set_title(
        f"Fine-Tuning Comparison: LLM Labels vs Ground Truth - {data_name.upper()}\\n"
        f"GLiNER Performance on Test Set",
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Number of Training Examples', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add value annotations at key points
    for i in [0, len(results_df)//2, -1]:
        if i < len(results_df):
            x = results_df.iloc[i]['n_requested']
            y_llm = results_df.iloc[i]['gliner_llm_ft_f1']
            y_gt = results_df.iloc[i]['gliner_gt_ft_f1']

            ax.annotate(f'{y_llm:.1f}%',
                       xy=(x, y_llm),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9,
                       alpha=0.7,
                       color=LLM_COLOR)

            ax.annotate(f'{y_gt:.1f}%',
                       xy=(x, y_gt),
                       xytext=(5, -15),
                       textcoords='offset points',
                       fontsize=9,
                       alpha=0.7,
                       color=GT_COLOR)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Plot saved: {output_file}")
    return output_file


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_confidence_gliner_llm_ft_f1(exp_config: Dict):
    """Run fine-tuning comparison experiment"""

    # Setup logging
    log_file = setup_experiment_logger(
        experiment_name=exp_config['experiment_name'],
        results_dir=exp_config['results_dir']
    )

    logging.info("="*80)
    logging.info("FINE-TUNING COMPARISON: LLM LABELS vs GROUND TRUTH")
    logging.info("="*80)
    logging.info(f"Device: {device}")
    logging.info(f"Data: {exp_config['data_name']}")
    logging.info(f"Strategy: {exp_config['strategy']}")
    logging.info(f"GLiNER: {exp_config['model_class'].__name__}")
    logging.info(f"LLM: {exp_config['llm_backend']} ({exp_config['llm_model_name']})")
    logging.info(f"Training sizes: {exp_config['subset_sizes']}")
    logging.info(f"Log file: {log_file}")

    # Load data
    logging.info("\n" + "="*80)
    logging.info("LOADING DATA")
    logging.info("="*80)

    with open(exp_config['train_data_path'], 'r') as f:
        train_data = json.load(f)

    with open(exp_config['test_data_path'], 'r') as f:
        test_data = json.load(f)

    with open(exp_config['labels_path'], 'r') as f:
        entity_types = json.load(f)

    logging.info(f"Train data: {len(train_data)} examples")
    logging.info(f"Test data: {len(test_data)} examples")
    logging.info(f"Entity types: {entity_types}")

    # Convert test data for evaluation
    converted_test = convert_raw_json_to_gliner_training(test_data)
    logging.info(f"Converted test data: {len(converted_test)} examples")

    # Load or generate GLiNER confidence data
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

    # Cleanup memory after confidence data generation
    logging.info("\n  Cleaning up memory after confidence data loading...")
    cleanup_memory()

    # Filter confidence data
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
            text = conf_ex['text']
            if text in train_lookup:
                has_ground_truth = len(train_lookup[text].get('entities', [])) > 0
                if has_ground_truth:
                    filtered_confidence_data.append(conf_ex)
        else:
            skipped_no_predictions += 1

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

    logging.info(f"\nSelected {len(examples)} worst confidence examples for training")

    # Load or create LLM labels
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

    logging.info(f"\nLLM labels loaded: {len(llm_labels)} examples")
    valid_count = sum(1 for label in llm_labels if len(label['entities']) > 0)
    logging.info(f"  Valid labels (with entities): {valid_count}")
    logging.info(f"  Empty labels (validation failed): {len(llm_labels) - valid_count}")

    # Cleanup memory after LLM label generation
    logging.info("\n  Cleaning up memory after LLM label loading...")
    cleanup_memory()

    # Prepare eval data for training (use first 100 test examples for early stopping)
    eval_data = convert_raw_json_to_gliner_training(test_data)
    logging.info(f"\nEval data for early stopping: {len(eval_data)} examples")

    # Run experiments
    logging.info("\n" + "="*80)
    logging.info("RUNNING FINE-TUNING EXPERIMENTS")
    logging.info("="*80)

    # Final cleanup before training loop
    logging.info("\n  Final memory cleanup before training...")
    cleanup_memory()

    results = {
        'n_requested': [],
        'gliner_llm_ft_f1': [],
        'gliner_gt_ft_f1': []
    }

    for n in exp_config['subset_sizes']:
        logging.info(f"\n{'='*80}")
        logging.info(f"TRAINING SIZE: {n}")
        logging.info(f"{'='*80}")

        # Prepare training data
        llm_subset = llm_labels[:n]
        gt_subset = examples[:n]

        # Convert to training format
        logging.info(f"\n  Preparing training data...")
        llm_train_data = prepare_llm_training_data(llm_subset)
        gt_train_data = convert_raw_json_to_gliner_training(gt_subset)

        logging.info(f"  LLM training data: {len(llm_train_data)} examples")
        logging.info(f"  GT training data: {len(gt_train_data)} examples")

        # Train on LLM labels
        logging.info(f"\n  {'='*60}")
        logging.info(f"  TRAINING ON LLM LABELS")
        logging.info(f"  {'='*60}")

        # Cleanup before training
        logging.info(f"  Cleaning up memory before LLM training...")
        cleanup_memory()

        adapter_llm = train_and_save_adapter(
            model_class=exp_config['model_class'],
            train_data=llm_train_data,
            eval_data=eval_data,
            lora_config=exp_config['lora_config'],
            training_config=exp_config['training_config'],
            adapter_name=f"llm_ft_{n}",
            results_dir=exp_config['results_dir']
        )

        # Evaluate LLM-FT
        logging.info(f"\n  Evaluating LLM-FT on test set...")
        logging.info(f"  Cleaning up memory before evaluation...")
        cleanup_memory()

        llm_ft_f1 = evaluate_on_test_set(
            model_class=exp_config['model_class'],
            adapter_path=adapter_llm,
            test_data=test_data,
            converted_test=converted_test,
            entity_types=entity_types,
            threshold=exp_config.get('gliner_threshold', 0.5)
        )
        logging.info(f"  LLM-FT F1: {llm_ft_f1:.2f}%")

        # Train on ground truth
        logging.info(f"\n  {'='*60}")
        logging.info(f"  TRAINING ON GROUND TRUTH")
        logging.info(f"  {'='*60}")

        # Cleanup before training
        logging.info(f"  Cleaning up memory before GT training...")
        cleanup_memory()

        adapter_gt = train_and_save_adapter(
            model_class=exp_config['model_class'],
            train_data=gt_train_data,
            eval_data=eval_data,
            lora_config=exp_config['lora_config'],
            training_config=exp_config['training_config'],
            adapter_name=f"gt_ft_{n}",
            results_dir=exp_config['results_dir']
        )

        # Evaluate GT-FT
        logging.info(f"\n  Evaluating GT-FT on test set...")
        logging.info(f"  Cleaning up memory before evaluation...")
        cleanup_memory()

        gt_ft_f1 = evaluate_on_test_set(
            model_class=exp_config['model_class'],
            adapter_path=adapter_gt,
            test_data=test_data,
            converted_test=converted_test,
            entity_types=entity_types,
            threshold=exp_config.get('gliner_threshold', 0.5)
        )
        logging.info(f"  GT-FT F1: {gt_ft_f1:.2f}%")

        # Store results
        results['n_requested'].append(n)
        results['gliner_llm_ft_f1'].append(llm_ft_f1)
        results['gliner_gt_ft_f1'].append(gt_ft_f1)

        # Comparison
        logging.info(f"\n  Results on Test Set:")
        logging.info(f"    LLM-FT: F1={llm_ft_f1:.2f}%")
        logging.info(f"    GT-FT:  F1={gt_ft_f1:.2f}%")
        winner = 'GT' if gt_ft_f1 > llm_ft_f1 else 'LLM' if llm_ft_f1 > gt_ft_f1 else 'TIE'
        logging.info(f"    Winner: {winner}")

    # Save results
    logging.info("\n" + "="*80)
    logging.info("SAVING RESULTS")
    logging.info("="*80)

    results_df = pd.DataFrame(results)
    logging.info("\n" + results_df.to_string(index=False))

    csv_file = save_results(results_df, exp_config)
    plot_file = generate_ft_comparison_plot(results_df, exp_config)

    logging.info(f"\nCSV:  {csv_file}")
    logging.info(f"Plot: {plot_file}")

    # Summary
    logging.info("\n" + "="*80)
    logging.info("SUMMARY")
    logging.info("="*80)
    logging.info(f"Best LLM-FT F1: {max(results['gliner_llm_ft_f1']):.2f}%")
    logging.info(f"Best GT-FT F1:  {max(results['gliner_gt_ft_f1']):.2f}%")
    logging.info(f"Adapters saved in: {exp_config['results_dir'] / 'adapters'}")
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
        'test_data_path': MIT_movies_NER_test_path,
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

        # Training
        'lora_config': DEFAULT_LORA_CONFIG,
        'training_config': {
            **DEFAULT_TRAINING_CONFIG,
            'save_strategy': 'no',  # Disable checkpoint saving
        },

        # Experiment
        'subset_sizes': [10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500],

        # Output
        'experiment_name': 'exp_confidence_gliner_llm_ft_f1',
        'results_dir': Path('/app/results2/exp_confidence_gliner_llm_ft_f1')
    }

    run_confidence_gliner_llm_ft_f1(exp_config)
