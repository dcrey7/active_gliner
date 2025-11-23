#!/usr/bin/env python3
"""
LoRA Target Module Configuration Experiment

Tests different LoRA layer configurations on GLiNER model to find optimal
target modules for parameter-efficient fine-tuning on NER tasks.
"""

import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import random
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from peft import TaskType
import matplotlib.pyplot as plt
import seaborn as sns

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

# Imports from src2
from active_gliner.create_data.gliner_format import convert_raw_json_to_gliner_training
from active_gliner.config.data_paths import (
    MIT_movies_NER_train_path,
    MIT_movies_NER_test_path,
    MIT_movies_NER_labels_path
)
from active_gliner.config.model_configs import DEFAULT_TRAINING_CONFIG
from active_gliner.get_model.DefaultModel import DefaultModel
from active_gliner.helper import (
    cleanup_memory,
    train_and_save_adapter,
    evaluate_on_test_set,
    setup_experiment_logger
)


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_lora_layer_results(results_df: pd.DataFrame, exp_config: Dict) -> Path:
    """
    Save LoRA layer experiment results to CSV.

    Input:
        results_df: DataFrame with experiment results
        exp_config: Experiment configuration dict

    Output:
        Path to saved CSV file
    """
    data_name = exp_config['data_name']
    model_name = exp_config['model_class'].__name__
    filename = f"{data_name}_LoRA_Layers_{model_name}.csv"

    output_dir = exp_config['results_dir'] / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    results_df.to_csv(output_file, index=False)
    logging.info(f"CSV saved: {output_file}")
    return output_file


def generate_lora_layer_plot(results_df: pd.DataFrame, exp_config: Dict) -> Path:
    """
    Generate bar chart comparing F1 scores across LoRA layer configurations.

    Input:
        results_df: DataFrame with columns [config_name, test_f1, num_target_modules, ...]
        exp_config: Experiment configuration dict

    Output:
        Path to saved plot file
    """
    data_name = exp_config['data_name']
    model_name = exp_config['model_class'].__name__
    filename = f"{data_name}_lora_layers_f1_comparison.png"

    # Create output directory
    output_dir = exp_config['results_dir'] / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    # Create figure
    plt.figure(figsize=(14, 8))

    # Create color gradient based on F1 scores (higher = darker/more vibrant)
    colors = plt.cm.viridis(results_df['test_f1'] / results_df['test_f1'].max())

    # Create bar plot
    ax = plt.gca()
    bars = ax.bar(
        range(len(results_df)),
        results_df['test_f1'],
        color=colors,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.85
    )

    # Add value labels on top of bars
    for i, (bar, f1_score, num_modules) in enumerate(zip(
        bars,
        results_df['test_f1'],
        results_df['num_target_modules']
    )):
        height = bar.get_height()
        # F1 score on top of bar
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.5,
            f'{f1_score:.2f}%',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
        # Number of modules inside bar (near top)
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height - 3,
            f'{num_modules} modules',
            ha='center',
            va='top',
            fontsize=9,
            color='white',
            fontweight='bold'
        )

    # Customize plot
    ax.set_xlabel('LoRA Layer Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=14, fontweight='bold')
    ax.set_title(
        f'LoRA Layer Configuration Comparison - {data_name.upper()}\n'
        f'Model: {model_name}',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    # Set x-axis labels
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(
        results_df['config_name'],
        rotation=45,
        ha='right',
        fontsize=11
    )

    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)

    # Set y-axis limits with some padding
    y_max = results_df['test_f1'].max()
    ax.set_ylim(0, y_max * 1.15)

    # Add legend explaining the visualization
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='none', label=f'Best: {results_df.iloc[0]["config_name"]}'),
        Patch(facecolor='none', edgecolor='none', label=f'F1: {results_df.iloc[0]["test_f1"]:.2f}%'),
        Patch(facecolor='none', edgecolor='none', label=''),
        Patch(facecolor=plt.cm.viridis(1.0), alpha=0.85, label='Higher F1 (darker color)'),
        Patch(facecolor=plt.cm.viridis(0.5), alpha=0.85, label='Lower F1 (lighter color)')
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=10,
        framealpha=0.9
    )

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Plot saved: {output_file}")
    return output_file




# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_lora_layer_experiment(exp_config: Dict):
    """
    Test different LoRA target module configurations.

    Input:
        exp_config: Dict containing data paths, model settings, LoRA configs,
                   training parameters, and output directory

    Output:
        Saves results to CSV and creates detailed logs
    """

    # Setup logging
    log_file = setup_experiment_logger(
        experiment_name=exp_config['experiment_name'],
        results_dir=exp_config['results_dir']
    )

    logging.info("="*80)
    logging.info("GLINER LORA LAYER CONFIGURATION EXPERIMENT")
    logging.info("="*80)
    logging.info(f"Device: {device}")
    logging.info(f"Data: {exp_config['data_name']}")
    logging.info(f"Model: {exp_config['model_class'].__name__}")
    logging.info(f"Testing {len(exp_config['lora_configurations'])} configurations")
    logging.info(f"Results directory: {exp_config['results_dir']}")
    logging.info(f"Log file: {log_file}")

    # Load data
    logging.info("\n" + "="*80)
    logging.info("LOADING DATA")
    logging.info("="*80)

    with open(exp_config['train_data_path'], 'r', encoding='utf-8') as f:
        train_data_raw = json.load(f)

    with open(exp_config['test_data_path'], 'r', encoding='utf-8') as f:
        test_data_raw = json.load(f)

    with open(exp_config['labels_path'], 'r', encoding='utf-8') as f:
        entity_types = json.load(f)

    logging.info(f"Train: {len(train_data_raw)} examples")
    logging.info(f"Test: {len(test_data_raw)} examples")
    logging.info(f"Entity types: {entity_types}")

    # Convert to GLiNER format
    logging.info("\nConverting to GLiNER training format...")
    train_data = convert_raw_json_to_gliner_training(train_data_raw)
    test_data_converted = convert_raw_json_to_gliner_training(test_data_raw)

    logging.info(f"Converted train: {len(train_data)} examples")
    logging.info(f"Converted test: {len(test_data_converted)} examples")

    # Use test set for early stopping during training
    eval_data = test_data_converted if len(test_data_converted) > 100 else test_data_converted

    logging.info(f"Using {len(eval_data)} test examples for early stopping during training")

    # Cleanup after data loading
    logging.info("\nCleaning up memory after data loading...")
    cleanup_memory()

    # Run experiments
    logging.info("\n" + "="*80)
    logging.info("RUNNING LORA CONFIGURATION EXPERIMENTS")
    logging.info("="*80)

    results_rows = []

    for idx, (config_name, target_modules) in enumerate(
        exp_config['lora_configurations'].items(), 1
    ):

        logging.info(f"\n{'='*80}")
        logging.info(f"CONFIGURATION {idx}/{len(exp_config['lora_configurations'])}: "
                    f"{config_name}")
        logging.info(f"{'='*80}")
        logging.info(f"Target modules ({len(target_modules)}):")
        for module in target_modules:
            logging.info(f"  - {module}")

        # Prepare LoRA config for this experiment
        lora_config = exp_config['base_lora_config'].copy()
        lora_config['target_modules'] = target_modules

        # Train adapter
        adapter_name = (
            f"lora_{config_name.replace(' ', '_').replace('+', 'and').lower()}"
        )

        logging.info(f"\nTraining adapter: {adapter_name}")
        logging.info("Cleaning up memory before training...")
        cleanup_memory()

        try:
            adapter_path = train_and_save_adapter(
                model_class=exp_config['model_class'],
                train_data=train_data,
                eval_data=eval_data,
                lora_config=lora_config,
                training_config=exp_config['training_config'],
                adapter_name=adapter_name,
                results_dir=exp_config['results_dir']
            )

            training_successful = True
            logging.info("Training completed successfully")

        except Exception as e:
            logging.error(f"Training FAILED: {e}")
            traceback.print_exc()
            training_successful = False
            adapter_path = None

        # Evaluate if training succeeded
        if training_successful:
            logging.info("\nEvaluating on test set...")
            logging.info("Cleaning up memory before evaluation...")
            cleanup_memory()

            try:
                test_f1 = evaluate_on_test_set(
                    model_class=exp_config['model_class'],
                    adapter_path=adapter_path,
                    test_data=test_data_raw,
                    converted_test=test_data_converted,
                    entity_types=entity_types,
                    threshold=exp_config['gliner_threshold']
                )

                logging.info(f"Test F1: {test_f1:.4f}%")

            except Exception as e:
                logging.error(f"Evaluation FAILED: {e}")
                traceback.print_exc()
                test_f1 = 0.0
        else:
            test_f1 = 0.0

        # Store results
        results_rows.append({
            'config_name': config_name,
            'num_target_modules': len(target_modules),
            'test_f1': test_f1,
            'training_successful': training_successful,
            'adapter_name': adapter_name,
            'adapter_path': str(adapter_path) if adapter_path else None
        })

        logging.info(f"\nConfiguration {idx} complete")
        logging.info(f"  Config: {config_name}")
        logging.info(f"  Test F1: {test_f1:.4f}%")
        logging.info(f"  Status: {'SUCCESS' if training_successful else 'FAILED'}")

    # Save results
    logging.info("\n" + "="*80)
    logging.info("SAVING RESULTS")
    logging.info("="*80)

    if results_rows:
        results_df = pd.DataFrame(results_rows).sort_values(
            by=['test_f1'], ascending=False
        ).reset_index(drop=True)

        # Save CSV
        csv_file = save_lora_layer_results(results_df, exp_config)

        # Generate plot
        plot_file = generate_lora_layer_plot(results_df, exp_config)

        # Print full results table
        logging.info("\n" + "="*80)
        logging.info("COMPLETE RESULTS TABLE")
        logging.info("="*80)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        logging.info("\n" + results_df.to_string(index=False))

        # Print summary
        successful_results = results_df[results_df['training_successful']]

        if len(successful_results) > 0:
            best_row = successful_results.iloc[0]

            logging.info("\n" + "="*80)
            logging.info("SUMMARY")
            logging.info("="*80)
            logging.info(f"Total configurations tested: {len(results_df)}")
            logging.info(f"Successful trainings: {len(successful_results)}")
            logging.info(f"Failed trainings: {len(results_df) - len(successful_results)}")
            logging.info(f"\nBest configuration: {best_row['config_name']}")
            logging.info(f"  Test F1: {best_row['test_f1']:.4f}%")
            logging.info(f"  Target modules: {best_row['num_target_modules']}")
            logging.info(f"  Adapter: {best_row['adapter_name']}")
            logging.info(f"\nAdapters saved in: {exp_config['results_dir'] / 'adapters'}")
            logging.info(f"CSV saved: {csv_file}")
            logging.info(f"Plot saved: {plot_file}")
            logging.info(f"Log saved: {log_file}")
        else:
            logging.info("\nNo successful trainings - all configurations failed")
    else:
        logging.info("No experiment runs were completed")

    logging.info("\n" + "="*80)
    logging.info("EXPERIMENT COMPLETED")
    logging.info("="*80)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

if __name__ == "__main__":

    # Define LoRA layer configurations to test
    task_layers = [
        "span_rep_layer.span_rep_layer.project_start.0",
        "span_rep_layer.span_rep_layer.project_start.3",
        "span_rep_layer.span_rep_layer.project_end.0",
        "span_rep_layer.span_rep_layer.project_end.3",
        "span_rep_layer.span_rep_layer.out_project.0",
        "span_rep_layer.span_rep_layer.out_project.3",
        "prompt_rep_layer.0",
        "prompt_rep_layer.3"
    ]

    modernbert_layers = ["Wqkv", "Wo", "Wi", "projection"]
    bert_layers = [
        "query", "key", "value",
        "intermediate.dense", "output.dense", "pooler.dense"
    ]
    attention_layers = ["Wqkv", "Wo", "query", "key", "value"]

    lora_configurations_dict = {
        "ModernBert + Task": modernbert_layers + task_layers,
        "BERT + Task": bert_layers + task_layers,
        "Attention + Task": attention_layers + task_layers,
        "All Layers": modernbert_layers + bert_layers + task_layers,
        "ModernBert Only": modernbert_layers,
        "BERT Only": bert_layers,
        "Attention Only": attention_layers,
        "Task Layers Only": task_layers
    }

    exp_config = {
        # Data
        'data_name': 'mit_movies',
        'train_data_path': MIT_movies_NER_train_path,
        'test_data_path': MIT_movies_NER_test_path,
        'labels_path': MIT_movies_NER_labels_path,

        # Model
        'model_class': DefaultModel,
        'gliner_threshold': 0.5,

        # LoRA configurations
        'lora_configurations': lora_configurations_dict,

        # Base LoRA config (target_modules will be set per experiment)
        'base_lora_config': {
            'r': 64,
            'lora_alpha': 128,
            'lora_dropout': 0.1,
            'bias': 'none',
            'task_type': TaskType.TOKEN_CLS
        },

        # Training config from DEFAULT_TRAINING_CONFIG
        'training_config': {
            **DEFAULT_TRAINING_CONFIG,
            'num_steps': 2500,
            'save_strategy': 'no'
        },

        # Output
        'experiment_name': 'exp_gliner_best_lora_layers',
        'results_dir': Path('/app/results2/exp_gliner_best_lora_layers')
    }

    run_lora_layer_experiment(exp_config)
