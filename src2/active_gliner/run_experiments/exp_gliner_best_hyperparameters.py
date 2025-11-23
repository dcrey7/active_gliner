#!/usr/bin/env python3
"""
GLiNER LoRA Hyperparameter Optimization with Optuna and MLflow

Uses Optuna to find optimal training hyperparameters for GLiNER with LoRA.
Tests different combinations of learning rates, LoRA architecture, batch sizes,
and regularization parameters on NER tasks. All trials are tracked in MLflow
for interactive comparison and visualization.

MLflow Usage:
    After running this experiment, view results in MLflow UI:

    1. Navigate to results directory:
       cd /app/results2/exp_gliner_best_hyperparameters

    2. Start MLflow UI (any available port):
       mlflow ui --backend-store-uri ./mlruns --port 5000

       Or use different port:
       mlflow ui --backend-store-uri ./mlruns --port 8080

    3. Open browser:
       http://localhost:5000  (or your chosen port)

    4. In MLflow UI:
       - Click on experiment: "gliner_mit_movies_hyperparameter_optimization"
       - Select multiple trial runs using checkboxes
       - Click "Compare" button to see side-by-side comparison
       - Use "Chart" tab to create interactive plots:
         * X-axis: any parameter (learning_rate, lora_r, batch_size, etc.)
         * Y-axis: test_f1
         * Group by: any other parameter
       - Use "Parallel Coordinates Plot" to visualize all parameters simultaneously
       - Filter and sort trials by any metric or parameter

    Results Location:
       - MLflow tracking data: results_dir/mlruns/
       - Trial adapters: results_dir/adapters/trial_XXX/
       - CSV results: results_dir/csv/
       - Logs: results_dir/logs/
       - Best parameters: results_dir/best_params.json
"""

import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import random
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict

import mlflow
import mlflow.pytorch
import optuna
import pandas as pd
import torch
from peft import TaskType

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
from active_gliner.config.model_configs import DEFAULT_LORA_CONFIG
from active_gliner.get_model.DefaultModel import DefaultModel
from active_gliner.helper import cleanup_memory, evaluate_on_test_set, setup_experiment_logger


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_optuna_results(study: optuna.Study, exp_config: Dict) -> Path:
    """
    Save Optuna study results to CSV.

    Input:
        study: Completed Optuna study
        exp_config: Experiment configuration dict

    Output:
        Path to saved CSV file
    """
    data_name = exp_config['data_name']
    model_name = exp_config['model_class'].__name__
    filename = f"{data_name}_Hyperparameters_{model_name}.csv"

    output_dir = exp_config['results_dir'] / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    # Extract trial data
    trials_data = []
    for trial in study.trials:
        trial_dict = {
            'trial_number': trial.number,
            'value': trial.value if trial.value is not None else 0.0,
            'state': trial.state.name,
            **trial.params
        }
        trials_data.append(trial_dict)

    results_df = pd.DataFrame(trials_data).sort_values(
        by=['value'], ascending=False
    ).reset_index(drop=True)

    results_df.to_csv(output_file, index=False)
    logging.info(f"CSV saved: {output_file}")
    return output_file


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def objective(trial, exp_config, train_data, eval_data, test_data_raw,
              test_data_converted, entity_types):
    """
    Optuna objective function for hyperparameter optimization.

    Input:
        trial: Optuna trial object
        exp_config: Experiment configuration dict
        train_data: Training data in GLiNER format
        eval_data: Evaluation data in GLiNER format (full test set)
        test_data_raw: Raw test data for evaluation
        test_data_converted: Test data in GLiNER format
        entity_types: List of entity types

    Output:
        Test F1 score (float)
    """
    trial_num = trial.number + 1
    trial_start = time.time()

    logging.info(f"\n{'='*80}")
    logging.info(f"TRIAL {trial_num}/{exp_config['n_trials']} STARTING")
    logging.info(f"{'='*80}")

    # Start MLflow run for this trial
    with mlflow.start_run(run_name=f"trial_{trial_num:03d}", nested=True):

        try:
            # Sample hyperparameters from search space
            params = {
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    *exp_config['search_space']['learning_rate_range'],
                    log=True
                ),
                'others_lr': trial.suggest_float(
                    'others_lr',
                    *exp_config['search_space']['others_lr_range'],
                    log=True
                ),
                'weight_decay': trial.suggest_float(
                    'weight_decay',
                    *exp_config['search_space']['weight_decay_range'],
                    log=True
                ),
                'others_weight_decay': trial.suggest_float(
                    'others_weight_decay',
                    *exp_config['search_space']['others_weight_decay_range'],
                    log=True
                ),
                'warmup_ratio': trial.suggest_float(
                    'warmup_ratio',
                    *exp_config['search_space']['warmup_ratio_range']
                ),
                'train_batch_size': trial.suggest_categorical(
                    'batch_size',
                    exp_config['search_space']['batch_size_choices']
                ),
                'lora_r': trial.suggest_categorical(
                    'lora_r',
                    exp_config['search_space']['lora_r_choices']
                ),
                'gradient_accumulation_steps': trial.suggest_categorical(
                    'gradient_accumulation_steps',
                    exp_config['search_space']['gradient_accumulation_choices']
                )
            }

            # Derived parameter: lora_alpha = 2 * lora_r
            lora_alpha = 2 * params['lora_r']
            params['lora_alpha'] = lora_alpha

            # Log parameters to MLflow
            mlflow.log_params(params)
            mlflow.log_params(exp_config['fixed_params'])
            mlflow.set_tag("trial_number", trial_num)
            mlflow.set_tag("optuna_trial_id", trial.number)

            logging.info(f"Configuration:")
            for key, value in params.items():
                logging.info(f"  {key}: {value}")
            effective_batch = params['train_batch_size'] * params['gradient_accumulation_steps']
            logging.info(f"  effective_batch_size: {effective_batch}")

            # Log effective batch size to MLflow
            mlflow.log_param("effective_batch_size", effective_batch)

            # Create trial-specific LoRA config
            trial_lora_config = {
                'r': params['lora_r'],
                'lora_alpha': lora_alpha,
                'target_modules': exp_config['target_modules'],
                'lora_dropout': exp_config['fixed_params']['lora_dropout'],
                'bias': 'none',
                'task_type': TaskType.TOKEN_CLS
            }

            # Create trial-specific training config
            trial_training_config = {
                'num_steps': exp_config['fixed_params']['max_steps_trial'],
                'save_strategy': 'no',
                'train_batch_size': params['train_batch_size'],
                'gradient_accumulation_steps': params['gradient_accumulation_steps'],
                'learning_rate': params['learning_rate'],
                'others_lr': params['others_lr'],
                'warmup_ratio': params['warmup_ratio'],
                'weight_decay': params['weight_decay'],
                'others_weight_decay': params['others_weight_decay'],
                'eval_steps': 100,
                'save_steps': 100,
                'logging_steps': 10,
                'max_grad_norm': exp_config['fixed_params']['max_grad_norm'],
                'focal_loss_alpha': exp_config['fixed_params']['focal_loss_alpha'],
                'focal_loss_gamma': exp_config['fixed_params']['focal_loss_gamma'],
                'patience': exp_config['fixed_params']['patience_trial']
            }

            # Prepare adapter path
            adapter_name = f"trial_{trial_num:03d}"
            adapter_path = exp_config['results_dir'] / "adapters" / adapter_name
            adapter_path.mkdir(parents=True, exist_ok=True)

            # Get data subsets for trial
            train_fraction = exp_config.get('trial_data_fraction', 1.0)
            if train_fraction < 1.0:
                subset_size = int(len(train_data) * train_fraction)
                trial_train_data = train_data[:subset_size]
            else:
                trial_train_data = train_data

            # Always use full test set for evaluation
            trial_eval_data = test_data_converted

            logging.info(f"\nTraining on {len(trial_train_data)} samples")
            logging.info(f"Validation on {len(trial_eval_data)} samples (full test set)")

            # Log dataset sizes to MLflow
            mlflow.log_metric("train_samples", len(trial_train_data))
            mlflow.log_metric("eval_samples", len(trial_eval_data))

            # Train with DefaultModel
            logging.info("\nCleaning up memory before training...")
            cleanup_memory()

            logging.info("Loading model for training...")
            model = DefaultModel(
                lora_config=trial_lora_config,
                training_config=trial_training_config
            )
            model.load_for_training()

            # Log GPU memory before training
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                mlflow.log_metric("gpu_memory_gb_start", gpu_memory_allocated)

            logging.info("Starting training...")
            model.fit(
                train_data=trial_train_data,
                eval_data=trial_eval_data,
                adapter_save_path=str(adapter_path)
            )

            training_successful = True
            logging.info("Training completed successfully")

            # Evaluate on test set
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

            # Log trial results
            trial_time = (time.time() - trial_start) / 60

            # Log metrics to MLflow
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("trial_time_minutes", trial_time)
            mlflow.log_metric("training_successful", 1 if training_successful else 0)

            # Log GPU memory after training
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                mlflow.log_metric("gpu_memory_gb_end", gpu_memory_allocated)

            # Log adapter path as tag
            mlflow.set_tag("adapter_path", str(adapter_path))

            logging.info(f"\nTrial {trial_num} complete")
            logging.info(f"  Test F1: {test_f1:.4f}%")
            logging.info(f"  Time: {trial_time:.1f} minutes")
            logging.info(f"  Status: {'SUCCESS' if training_successful else 'FAILED'}")

            # Cleanup
            cleanup_memory()

            return test_f1

        except Exception as e:
            logging.error(f"Trial {trial_num} FAILED: {e}")
            traceback.print_exc()

            # Log failure to MLflow
            mlflow.log_metric("test_f1", 0.0)
            mlflow.log_metric("training_successful", 0)
            mlflow.set_tag("error", str(e))

            cleanup_memory()
            return 0.0


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_hyperparameter_experiment(exp_config: Dict):
    """
    Run Optuna hyperparameter optimization experiment.

    Input:
        exp_config: Dict containing data paths, model settings, search space,
                   fixed params, and output directory

    Output:
        Saves results to CSV, creates logs, and saves best parameters
    """

    # Setup logging
    log_file = setup_experiment_logger(
        experiment_name=exp_config['experiment_name'],
        results_dir=exp_config['results_dir']
    )

    logging.info("="*80)
    logging.info("GLINER HYPERPARAMETER OPTIMIZATION EXPERIMENT")
    logging.info("="*80)
    logging.info(f"Device: {device}")
    logging.info(f"Data: {exp_config['data_name']}")
    logging.info(f"Model: {exp_config['model_class'].__name__}")
    logging.info(f"Number of trials: {exp_config['n_trials']}")
    logging.info(f"Results directory: {exp_config['results_dir']}")
    logging.info(f"Log file: {log_file}")

    # Setup MLflow
    mlflow_dir = exp_config['results_dir'] / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")

    experiment_name = f"gliner_{exp_config['data_name']}_hyperparameter_optimization"
    mlflow.set_experiment(experiment_name)

    logging.info(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")
    logging.info(f"MLflow experiment: {experiment_name}")

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

    # Use full test set for early stopping during training
    eval_data = test_data_converted

    logging.info(f"Using {len(eval_data)} test examples (full test set) for early stopping during training")

    # Cleanup after data loading
    logging.info("\nCleaning up memory after data loading...")
    cleanup_memory()

    # Run Optuna optimization
    logging.info("\n" + "="*80)
    logging.info("RUNNING OPTUNA OPTIMIZATION")
    logging.info("="*80)

    logging.info("\nSearch space:")
    for key, value in exp_config['search_space'].items():
        logging.info(f"  {key}: {value}")

    logging.info("\nFixed parameters:")
    for key, value in exp_config['fixed_params'].items():
        logging.info(f"  {key}: {value}")

    # Start parent MLflow run for entire optimization
    with mlflow.start_run(run_name="optuna_optimization"):

        # Log experiment config to parent run
        mlflow.log_param("n_trials", exp_config['n_trials'])
        mlflow.log_param("data_name", exp_config['data_name'])
        mlflow.log_param("model_class", exp_config['model_class'].__name__)
        mlflow.log_param("trial_data_fraction", exp_config.get('trial_data_fraction', 1.0))
        mlflow.set_tag("experiment_type", "hyperparameter_optimization")

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )

        # Progress callback
        def progress_callback(study, trial):
            if len(study.trials) > 0:
                logging.info(f"\n{'='*80}")
                logging.info(f"PROGRESS UPDATE")
                logging.info(f"{'='*80}")
                logging.info(f"Completed trials: {len(study.trials)}/{exp_config['n_trials']}")
                if study.best_value is not None:
                    logging.info(f"Best F1 so far: {study.best_value:.4f}%")
                    logging.info(f"Best params: {study.best_params}")

                    # Log progress to MLflow parent run
                    mlflow.log_metric("best_f1_so_far", study.best_value, step=len(study.trials))
                    mlflow.log_metric("completed_trials", len(study.trials))

                logging.info(f"{'='*80}\n")

        # Run optimization
        study.optimize(
            lambda trial: objective(
                trial, exp_config, train_data, eval_data,
                test_data_raw, test_data_converted, entity_types
            ),
            n_trials=exp_config['n_trials'],
            callbacks=[progress_callback]
        )

        # Save results
        logging.info("\n" + "="*80)
        logging.info("SAVING RESULTS")
        logging.info("="*80)

        # Save CSV
        csv_file = save_optuna_results(study, exp_config)

        # Print full results table
        logging.info("\n" + "="*80)
        logging.info("COMPLETE RESULTS TABLE")
        logging.info("="*80)

        trials_data = []
        for trial in study.trials:
            trial_dict = {
                'trial': trial.number,
                'f1_score': trial.value if trial.value is not None else 0.0,
                'state': trial.state.name,
                **trial.params
            }
            trials_data.append(trial_dict)

        results_df = pd.DataFrame(trials_data).sort_values(
            by=['f1_score'], ascending=False
        ).reset_index(drop=True)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        logging.info("\n" + results_df.to_string(index=False))

        # Print summary and log to MLflow
        if study.best_value is not None:
            best_params = study.best_params.copy()
            best_params['lora_alpha'] = 2 * best_params['lora_r']

            # Log final summary to MLflow parent run
            mlflow.log_metric("final_best_f1", study.best_value)
            mlflow.log_metric("total_trials", len(study.trials))
            mlflow.log_metric("completed_trials", len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
            mlflow.log_metric("failed_trials", len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]))

            # Log best parameters as params (with prefix to avoid conflicts)
            for key, value in best_params.items():
                mlflow.log_param(f"best_{key}", value)

            logging.info("\n" + "="*80)
            logging.info("SUMMARY")
            logging.info("="*80)
            logging.info(f"Total trials: {len(study.trials)}")
            logging.info(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
            logging.info(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
            logging.info(f"\nBest trial: {study.best_trial.number}")
            logging.info(f"  Test F1: {study.best_value:.4f}%")
            logging.info(f"\nBest parameters:")
            for key, value in best_params.items():
                logging.info(f"  {key}: {value}")
            logging.info(f"\nAdapters saved in: {exp_config['results_dir'] / 'adapters'}")
            logging.info(f"CSV saved: {csv_file}")
            logging.info(f"Log saved: {log_file}")

            # Save best parameters to JSON
            best_params_file = exp_config['results_dir'] / "best_params.json"
            with open(best_params_file, 'w') as f:
                json.dump(best_params, f, indent=2)
            logging.info(f"Best parameters saved: {best_params_file}")

            # Log artifacts to MLflow
            mlflow.log_artifact(str(best_params_file))
            mlflow.log_artifact(str(csv_file))

        else:
            logging.info("\nNo successful trials - all trials failed")
            mlflow.log_metric("final_best_f1", 0.0)
            mlflow.set_tag("status", "all_trials_failed")

        logging.info("\n" + "="*80)
        logging.info("EXPERIMENT COMPLETED")
        logging.info("="*80)
        logging.info(f"\nView MLflow UI with: mlflow ui --backend-store-uri {mlflow_dir} --port 5000")
        logging.info(f"Or use different port: mlflow ui --backend-store-uri {mlflow_dir} --port 8080")


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

if __name__ == "__main__":

    # Target modules from DEFAULT_LORA_CONFIG
    target_modules = DEFAULT_LORA_CONFIG['target_modules']

    exp_config = {
        # Data
        'data_name': 'mit_movies',
        'train_data_path': MIT_movies_NER_train_path,
        'test_data_path': MIT_movies_NER_test_path,
        'labels_path': MIT_movies_NER_labels_path,

        # Model
        'model_class': DefaultModel,
        'gliner_threshold': 0.5,
        'target_modules': target_modules,

        # Optuna settings
        'n_trials': 20,
        'trial_data_fraction': 1.0,  # Use 100% data for trials (use <1.0 for faster trials)

        # Search space
        'search_space': {
            'learning_rate_range': (5e-6, 5e-3),
            'others_lr_range': (5e-6, 5e-3),
            'weight_decay_range': (1e-5, 0.1),
            'others_weight_decay_range': (1e-5, 0.1),
            'warmup_ratio_range': (0.05, 0.2),
            'batch_size_choices': [2, 4, 8, 12, 16],
            'lora_r_choices': [8, 16, 32, 64],
            'gradient_accumulation_choices': [1, 2, 4]
        },

        # Fixed parameters
        'fixed_params': {
            'lora_dropout': 0.1,
            'focal_loss_alpha': 0.75,  # Disabled (0 = no focal loss)
            'focal_loss_gamma': 1.0,  # Disabled (0 = no focal loss)
            'max_grad_norm': 1.0,
            'max_steps_trial': 2500,
            'patience_trial': 7
        },

        # Output
        'experiment_name': 'exp_gliner_best_hyperparameters',
        'results_dir': Path('/app/results2/exp_gliner_best_hyperparameters')
    }

    run_hyperparameter_experiment(exp_config)
