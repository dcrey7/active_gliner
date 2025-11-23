#!/usr/bin/env python3
"""
Fine-tuning Comparison: GLiNER trained on LLM labels vs Ground Truth labels and mixing ratios

Swappable experiment framework - change exp_config to swap data/model/LLM
"""

import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import random
import sys
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

warnings.filterwarnings('ignore')
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
    load_or_create_gliner_confidence_data,
    cleanup_memory,
    prepare_llm_training_data,
    load_or_create_llm_labels,
    train_and_save_adapter,
    evaluate_on_test_set,
    save_results,
    generate_plot,
    setup_experiment_logger
)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_confidence_mixed_ft_f1(exp_config: Dict):
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


    results_rows = []

    for n in exp_config['subset_sizes']:
        available_examples = min(n, len(llm_labels), len(examples))
        if available_examples == 0:
            logging.info(f"\nSkipping training size {n}: no overlapping data available.")
            continue

        for mixing_ratio in exp_config['mixing_ratios']:
            logging.info(f"\n{'='*80}")
            logging.info(f"TRAINING SIZE REQUESTED: {n}")
            if available_examples != n:
                logging.info(f"Using {available_examples} examples due to data availability")
            logging.info(f"{'='*80}")

            logging.info(f"\n{'='*80}")
            logging.info(f"MIXING RATIO: {mixing_ratio}% Ground truth labels, {100-mixing_ratio}% LLM labels ")
            logging.info(f"{'='*80}")

            # Prepare training data
            llm_subset = llm_labels[:available_examples]
            gt_subset = examples[:available_examples]

            # Convert to training format
            logging.info(f"\n  Preparing training data...")
            llm_train_data_raw = prepare_llm_training_data(llm_subset)
            gt_train_data_raw = convert_raw_json_to_gliner_training(gt_subset)

            gt_fraction = mixing_ratio / 100
            gt_count = int(round(available_examples * gt_fraction))
            llm_count = available_examples - gt_count

            gt_train_data = gt_train_data_raw[:gt_count]
            llm_train_data = llm_train_data_raw[:llm_count]

            logging.info(f"  LLM training data: {len(llm_train_data)} examples")
            logging.info(f"  GT training data: {len(gt_train_data)} examples")

            train_data = llm_train_data + gt_train_data
            logging.info(f"  Total mixed training examples: {len(train_data)}")

            logging.info(f"\n  {'='*60}")
            logging.info(f"  TRAINING MIXED DATASET")
            logging.info(f"  {'='*60}")

            # Cleanup before training
            logging.info(f"  Cleaning up memory before training...")
            cleanup_memory()

            adapter_name = f"mixed_ft_{n}_{mixing_ratio}_gt_{100-mixing_ratio}_llm"
            adapter_path = train_and_save_adapter(
                model_class=exp_config['model_class'],
                train_data=train_data,
                eval_data=eval_data,
                lora_config=exp_config['lora_config'],
                training_config=exp_config['training_config'],
                adapter_name=adapter_name,
                results_dir=exp_config['results_dir']
            )

            logging.info(f"\n  Evaluating mixed adapter on test set...")
            logging.info(f"  Cleaning up memory before evaluation...")
            cleanup_memory()

            mixed_ft_f1 = evaluate_on_test_set(
                model_class=exp_config['model_class'],
                adapter_path=adapter_path,
                test_data=test_data,
                converted_test=converted_test,
                entity_types=entity_types,
                threshold=exp_config.get('gliner_threshold', 0.5)
            )
            logging.info(f"  MIXED FT {mixing_ratio} GT and {100-mixing_ratio} LLM labels is F1: {mixed_ft_f1:.2f}%")

            results_rows.append({
                'n_requested': n,
                'n_actual': available_examples,
                'mixing_ratio_pct': mixing_ratio,
                'n_llm_examples': llm_count,
                'n_gt_examples': gt_count,
                'mixed_ft_f1': mixed_ft_f1,
                'adapter_name': adapter_name,
                'adapter_path': str(adapter_path)
            })

    logging.info("\n" + "="*80)
    logging.info("SAVING RESULTS")
    logging.info("="*80)

    if results_rows:
        results_df = pd.DataFrame(results_rows).sort_values(
            by=['n_requested', 'mixing_ratio_pct', 'n_actual']
        ).reset_index(drop=True)
        logging.info("\nDetailed results per run:\n")
        logging.info(results_df.to_string(index=False))

        ratio_col_map = {
            ratio: f"gliner_ft_{ratio}gt_{100 - ratio}llm_f1"
            for ratio in exp_config['mixing_ratios']
        }

        pivot_df = (
            results_df
            .pivot_table(
                index='n_actual',
                columns='mixing_ratio_pct',
                values='mixed_ft_f1',
                aggfunc='max'
            )
            .rename(columns=ratio_col_map)
            .reset_index()
            .rename(columns={'n_actual': 'no_worst_examples'})
            .sort_values('no_worst_examples')
        )

        for ratio, col_name in ratio_col_map.items():
            if col_name not in pivot_df.columns:
                pivot_df[col_name] = pd.NA

        mix_counts = (
            results_df
            .sort_values('mixing_ratio_pct')
            .groupby('n_actual')
            .apply(
                lambda grp: json.dumps([
                    {
                        'mixing_ratio_pct': int(row['mixing_ratio_pct']),
                        'gt_examples': int(row['n_gt_examples']),
                        'llm_examples': int(row['n_llm_examples'])
                    }
                    for _, row in grp.iterrows()
                ])
            )
            .rename('mix_counts')
        )

        pivot_df = pivot_df.merge(
            mix_counts,
            left_on='no_worst_examples',
            right_index=True,
            how='left'
        )

        ordered_columns = ['no_worst_examples'] + [
            ratio_col_map[ratio] for ratio in exp_config['mixing_ratios']
        ] + ['mix_counts']
        pivot_df = pivot_df[ordered_columns]

        logging.info("\nCSV summary (one row per subset size):\n")
        logging.info(pivot_df.to_string(index=False))

        csv_file = save_results(pivot_df, exp_config)
        plot_file = generate_plot(results_df, exp_config)

        logging.info(f"\nCSV:  {csv_file}")
        logging.info(f"Plot: {plot_file}")

        best_idx = results_df['mixed_ft_f1'].idxmax()
        best_row = results_df.loc[best_idx]

        logging.info("\n" + "="*80)
        logging.info("SUMMARY")
        logging.info("="*80)
        logging.info(f"Best mixed F1: {best_row['mixed_ft_f1']:.2f}%")
        logging.info(f"  Training size requested: {best_row['n_requested']}")
        logging.info(f"  Actual examples used:    {best_row['n_actual']}")
        logging.info(f"  Mixing ratio:            {best_row['mixing_ratio_pct']}% GT / {100 - best_row['mixing_ratio_pct']}% LLM")
        logging.info(f"Adapters saved in: {exp_config['results_dir'] / 'adapters'}")
    else:
        logging.info("No experiment runs were completed, nothing to save.")

    logging.info("\nCompleted!")



# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    strategy = 'min'  # Options: 'mse', 'min', 'mnlp', 'avg'

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
        'mixing_ratios': [0, 25, 50, 75, 100],

        # Output
        'experiment_name': 'exp_confidence_mixed_ft_f1',
        'results_dir': Path(f'/app/results2/exp_confidence_mixed_ft_f1_{strategy}_strategy/'),
    }

    run_confidence_mixed_ft_f1(exp_config)
