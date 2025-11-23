#!/usr/bin/env python3
"""
Active Learning Strategy Comparison Experiment

Compares selection strategies (MSE, Least Confidence, MNLP, Random)
by fine-tuning GLiNER on examples ranked by each strategy.

Goal: Show which strategy reaches target F1 with fewer labeled examples.
"""

import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List

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
    cleanup_memory,
    train_and_save_adapter,
    evaluate_on_test_set,
    setup_experiment_logger,
    load_or_create_strategy_data,
    filter_examples_with_predictions,
    generate_active_learning_plot
)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_active_learning_strategies(exp_config: Dict):
    """
    Run active learning strategy comparison experiment.

    For each strategy, fine-tune on top-N examples and evaluate.
    """
    # Setup logging
    log_file = setup_experiment_logger(
        experiment_name=exp_config['experiment_name'],
        results_dir=exp_config['results_dir']
    )

    logging.info("=" * 80)
    logging.info("ACTIVE LEARNING STRATEGY COMPARISON")
    logging.info("=" * 80)
    logging.info(f"Device: {device}")
    logging.info(f"Data: {exp_config['data_name']}")
    logging.info(f"GLiNER: {exp_config['model_class'].__name__}")
    logging.info(f"Strategies: {exp_config['strategies']}")
    logging.info(f"Subset sizes: {exp_config['subset_sizes']}")
    logging.info(f"Log file: {log_file}")

    # Load data
    logging.info("\n" + "=" * 80)
    logging.info("LOADING DATA")
    logging.info("=" * 80)

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

    # Prepare eval data for early stopping
    eval_data = convert_raw_json_to_gliner_training(test_data)
    logging.info(f"Eval data for early stopping: {len(eval_data)} examples")

    # Load or create strategy-sorted data
    logging.info("\n" + "=" * 80)
    logging.info("LOADING/GENERATING STRATEGY-SORTED DATA")
    logging.info("=" * 80)

    strategy_data = load_or_create_strategy_data(
        model_class=exp_config['model_class'],
        adapter_path=exp_config.get('model_adapter_path'),
        train_data=train_data,
        entity_types=entity_types,
        base_confidence_path=exp_config['confidence_data_path'],
        threshold=exp_config.get('gliner_threshold', 0.5)
    )

    # Cleanup after loading
    cleanup_memory()

    # Filter examples with predictions
    logging.info("\n" + "=" * 80)
    logging.info("FILTERING EXAMPLES WITH PREDICTIONS")
    logging.info("=" * 80)

    filtered_data = filter_examples_with_predictions(
        strategy_data=strategy_data,
        train_data=train_data,
        max_examples=exp_config['max_examples']
    )

    # Log filtered counts
    for strategy, examples in filtered_data.items():
        logging.info(f"  {strategy}: {len(examples)} filtered examples")

    # Run experiments
    logging.info("\n" + "=" * 80)
    logging.info("RUNNING FINE-TUNING EXPERIMENTS")
    logging.info("=" * 80)

    results_rows = []

    for strategy in exp_config['strategies']:
        logging.info(f"\n{'=' * 80}")
        logging.info(f"STRATEGY: {strategy.upper()}")
        logging.info(f"{'=' * 80}")

        strategy_examples = filtered_data.get(strategy, [])
        if not strategy_examples:
            logging.info(f"No examples for strategy {strategy}, skipping...")
            continue

        for n in exp_config['subset_sizes']:
            available = min(n, len(strategy_examples))
            if available == 0:
                logging.info(f"  Skipping size {n}: no data available")
                continue

            logging.info(f"\n  Training size: {n} (using {available} examples)")

            # Get top-N examples for this strategy
            subset = strategy_examples[:available]

            # Extract ground truth for training
            gt_examples = [ex['ground_truth'] for ex in subset]
            train_data_subset = convert_raw_json_to_gliner_training(gt_examples)

            logging.info(f"  Training data: {len(train_data_subset)} examples")

            # Cleanup before training
            cleanup_memory()

            # Train adapter
            adapter_name = f"{strategy}_{available}"
            adapter_path = train_and_save_adapter(
                model_class=exp_config['model_class'],
                train_data=train_data_subset,
                eval_data=eval_data,
                lora_config=exp_config['lora_config'],
                training_config=exp_config['training_config'],
                adapter_name=adapter_name,
                results_dir=exp_config['results_dir']
            )

            # Evaluate
            logging.info(f"  Evaluating on test set...")
            cleanup_memory()

            f1 = evaluate_on_test_set(
                model_class=exp_config['model_class'],
                adapter_path=adapter_path,
                test_data=test_data,
                converted_test=converted_test,
                entity_types=entity_types,
                threshold=exp_config.get('gliner_threshold', 0.5)
            )

            logging.info(f"  {strategy.upper()} @ {available} examples: F1 = {f1:.2f}%")

            results_rows.append({
                'strategy': strategy,
                'subset_size': available,
                'f1': f1,
                'adapter_name': adapter_name,
                'adapter_path': str(adapter_path)
            })

            # Cleanup after evaluation
            cleanup_memory()

    # Save results
    logging.info("\n" + "=" * 80)
    logging.info("SAVING RESULTS")
    logging.info("=" * 80)

    if results_rows:
        results_df = pd.DataFrame(results_rows)

        # Log detailed results
        logging.info("\nDetailed results:\n")
        logging.info(results_df.to_string(index=False))

        # Create pivot table for easy comparison
        pivot_df = results_df.pivot_table(
            index='subset_size',
            columns='strategy',
            values='f1',
            aggfunc='max'
        ).reset_index()

        # Rename columns
        pivot_df.columns = ['n_examples'] + [f'{s}_f1' for s in pivot_df.columns[1:]]

        logging.info("\nPivot table (F1 by strategy and size):\n")
        logging.info(pivot_df.to_string(index=False))

        # Save CSV
        csv_dir = exp_config['results_dir'] / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        detailed_csv = csv_dir / f"{exp_config['data_name']}_active_learning_detailed.csv"
        results_df.to_csv(detailed_csv, index=False)
        logging.info(f"\nDetailed CSV: {detailed_csv}")

        pivot_csv = csv_dir / f"{exp_config['data_name']}_active_learning_pivot.csv"
        pivot_df.to_csv(pivot_csv, index=False)
        logging.info(f"Pivot CSV: {pivot_csv}")

        # Generate plot
        plot_file = generate_active_learning_plot(results_df, exp_config)
        logging.info(f"Plot: {plot_file}")

        # Summary statistics
        logging.info("\n" + "=" * 80)
        logging.info("SUMMARY")
        logging.info("=" * 80)

        for strategy in exp_config['strategies']:
            strategy_df = results_df[results_df['strategy'] == strategy]
            if not strategy_df.empty:
                best_row = strategy_df.loc[strategy_df['f1'].idxmax()]
                logging.info(f"\n{strategy.upper()}:")
                logging.info(f"  Best F1: {best_row['f1']:.2f}% at {best_row['subset_size']} examples")

                # Find minimum examples to reach target F1
                target_f1 = exp_config.get('target_f1', 80.0)
                above_target = strategy_df[strategy_df['f1'] >= target_f1]
                if not above_target.empty:
                    min_examples = above_target['subset_size'].min()
                    logging.info(f"  Reaches {target_f1}% F1 at: {min_examples} examples")
                else:
                    logging.info(f"  Does not reach {target_f1}% F1")

        logging.info(f"\nAdapters saved in: {exp_config['results_dir'] / 'adapters'}")

    else:
        logging.info("No experiment runs completed.")

    logging.info("\nCompleted!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    exp_config = {
        # Data
        'data_name': 'mit_movies',
        'train_data_path': MIT_movies_NER_train_path,
        'test_data_path': MIT_movies_NER_test_path,
        'labels_path': MIT_movies_NER_labels_path,
        'confidence_data_path': '/app/data/experiment_data',
        'max_examples': 2500,

        # GLiNER model
        'model_class': DefaultModel,
        'model_adapter_path': None,
        'gliner_threshold': 0.5,

        # Training
        'lora_config': DEFAULT_LORA_CONFIG,
        'training_config': {
            **DEFAULT_TRAINING_CONFIG,
            'save_strategy': 'no',
        },

        # Experiment
        'strategies': ['mse', 'min', 'mnlp', 'random'],
        'subset_sizes': [10, 50, 100, 200, 400, 600, 800, 1000, 1250, 1500, 1750, 2000, 2250, 2500],
        'target_f1': 80.0,

        # Output
        'experiment_name': 'exp_active_learning_confidence_strategies',
        'results_dir': Path('/app/results2/exp_active_learning_confidence_strategies')
    }

    run_active_learning_strategies(exp_config)
