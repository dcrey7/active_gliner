#!/usr/bin/env python3
"""
Confidence Analysis Script 2: Fine-tuning Performance Analysis
Tests GLiNER fine-tuned on LLM labels vs GT labels of worst confidence examples
Evaluates fine-tuned models on FULL MIT test set

Uses new abstractions:
- GLONER for model initialization and loading
- create_label_generator for LLM labeling with caching
- train_lora_model for training
- NERValidator for validation with reporting
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Add src path
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.append(src_path)

# Import with full paths (no __init__ dependencies!)
from config.settings import Settings
from utils.logging import setup_logging, get_logger
from utils.reproducibility import set_all_seeds
from utils.device import setup_device
from utils.memory import cleanup_memory
from data.loader import load_mit_dataset
from data.validator import NERValidator
from evaluation.evaluator import enhanced_evaluate
from generation.label_generator import create_label_generator
from training.trainer import train_lora_model
from models.gloner import GLONER

# Constants
GLOBAL_SEED = 42


def main():
    """Confidence Analysis: Fine-tuning Performance on Worst Examples"""

    # ===============================================================================
    # Setup and Configuration
    # ===============================================================================

    settings = Settings()
    settings.setup()
    logger = setup_logging(log_dir=str(settings.logs_dir))
    set_all_seeds(seed=GLOBAL_SEED, logger=logger)
    device = setup_device(logger=logger)

    # LLM Configuration
    LLM_BACKEND = "ollama"  # ollama, mistral, cerebras
    LLM_MODEL = "gemma3:12b"
    USE_STRUCTURED = False  # ollama doesn't support structured

    # Load FULL test data for evaluation
    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file

    if not (test_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Test data or labels file not found!")

    test_data, entity_types = load_mit_dataset(str(test_data_path), str(labels_path), "test")
    logger.info(f"Loaded FULL test data: {len(test_data)} examples, {len(entity_types)} entity types")

    # Load pre-saved low confidence examples for training
    logger.info("Loading pre-saved low confidence examples...")
    low_conf_file = os.path.join(os.path.dirname(__file__), '../results/high_mse_2500_examples.json')
    with open(low_conf_file, 'r') as f:
        low_n = json.load(f)
    logger.info(f"Loaded {len(low_n)} low confidence examples for training")

    # Initialize LLM label generator with disk cache
    logger.info("Initializing LLM label generator with disk cache...")
    label_generator = create_label_generator(
        backend_type=LLM_BACKEND,
        model_name=LLM_MODEL,
        cache_type='disk',  # Persistent disk cache
        use_structured_output=USE_STRUCTURED
    )
    logger.info(f"LLM Labeler: {LLM_BACKEND} - {LLM_MODEL}")

    # Initialize validator
    validator = NERValidator(entity_types=entity_types, logger=logger)

    # ===============================================================================
    # Training Configuration
    # ===============================================================================

    training_config = {
        'num_steps': 1000,
        'train_batch_size': 8,
        'gradient_accumulation_steps': 1,
        'learning_rate': 0.00021008343694753508,
        'others_lr': 0.00021008343694753508,
        'warmup_ratio': 0.07064690788186724,
        'eval_steps': 100,
        'save_steps': 100,
        'logging_steps': 10,
        'max_grad_norm': 1,
        'weight_decay': 0.020216630535603918,
        'others_weight_decay': 0.020216630535603918,
        'focal_loss_alpha': 0.75,
        'focal_loss_gamma': 1.0,
        'patience': 3
    }

    logger.info("Training Configuration:")
    for key, value in training_config.items():
        logger.info(f"   {key}: {value}")

    # ===============================================================================
    # Experiment Parameters
    # ===============================================================================

    subset_sizes = [2, 4, 8, 10]  # Small sizes for testing

    # Results storage
    results = {
        'no_worst_examples': [],
        'gliner_ft_llm_f1': [],
        'gliner_ft_gt_f1': [],
        'confidence': [],
        'avg_entities': [],
        'avg_input_tokens': [],
        'avg_output_tokens': []
    }

    total_iterations = len(subset_sizes) * 2  # 2 experiments per subset (LLM + GT)
    logger.info(f"\nExperiment Overview:")
    logger.info(f"   Subset sizes to test: {subset_sizes}")
    logger.info(f"   Total training experiments: {total_iterations}")
    logger.info(f"   Evaluation dataset: FULL test set ({len(test_data)} examples)")

    # ===============================================================================
    # Main Experiment Loop
    # ===============================================================================

    logger.info(f"\nStarting Fine-tuning Analysis...")
    logger.info("-" * 60)

    for n_examples in tqdm(subset_sizes, desc="Training Experiments", position=0):
        logger.info(f"\nTraining on {n_examples} worst confidence examples")

        # Get subset for training
        train_subset = low_n[:n_examples]
        logger.info(f"Training subset size: {len(train_subset)} examples")

        # ===============================================================================
        # Generate LLM Labels with Caching
        # ===============================================================================

        logger.info(f"Generating LLM labels for {n_examples} examples (with disk caching)...")

        # Generate labels (automatically uses disk cache)
        llm_gen_results = label_generator.generate(
            low_n_examples=train_subset,
            num_samples=n_examples,
            entity_types=entity_types
        )

        llm_labeled_data = llm_gen_results['all_labels']

        # Validate using validator (additional validation layer)
        logger.info("Validating LLM labels...")
        llm_labeled_data, llm_report = validator.validate(llm_labeled_data, strict=True)
        logger.info(f"Validation: {len(llm_labeled_data)} valid examples")
        if logger:
            logger.info(llm_report.summary())

        # Calculate metrics
        if len(llm_labeled_data) > 0:
            avg_entities = sum(len(ex['ner']) for ex in llm_labeled_data) / len(llm_labeled_data)
            avg_input_tokens = llm_gen_results['total_input_tokens'] / len(llm_labeled_data) if len(llm_labeled_data) > 0 else 0
            avg_output_tokens = llm_gen_results['total_output_tokens'] / len(llm_labeled_data) if len(llm_labeled_data) > 0 else 0
        else:
            avg_entities = 0.0
            avg_input_tokens = 0.0
            avg_output_tokens = 0.0

        logger.info(f"Generated: {len(llm_labeled_data)} examples, avg entities: {avg_entities:.1f}")
        logger.info(f"Token usage: {avg_input_tokens:.0f} in, {avg_output_tokens:.0f} out per example")

        # ===============================================================================
        # Training Phase 1: GLiNER FT on LLM Labels
        # ===============================================================================

        if len(llm_labeled_data) > 0:
            logger.info(f"\nTraining GLiNER on LLM labels ({n_examples} examples)")

            # Define adapter save path
            models_dir = os.path.join(os.path.dirname(__file__), '../models')
            llm_adapter_path = os.path.join(models_dir, f"confidence_llm_model_{n_examples}")
            os.makedirs(models_dir, exist_ok=True)

            # Initialize model with LoRA using GLONER
            model = GLONER.default(logger=logger)
            model.to(device)

            # Train the model on LLM labels
            train_lora_model(
                model=model,
                train_data=llm_labeled_data,
                eval_data=test_data,  # Small eval subset to speed up training
                training_config=training_config,
                adapter_save_path=llm_adapter_path,
                logger=logger
            )

            # Cleanup training model
            del model
            cleanup_memory()

            # ===============================================================================
            # Evaluation Phase 1: GLiNER FT on LLM Labels
            # ===============================================================================

            logger.info(f"Evaluating GLiNER FT (LLM labels) on FULL test set...")

            # Load model with trained adapter using GLONER
            eval_model = GLONER.load_with_adapter(llm_adapter_path, logger=logger)

            # Enhanced evaluation on FULL test set
            import torch
            with torch.no_grad():
                llm_ft_results = enhanced_evaluate(
                    eval_model, test_data, entity_types,
                    threshold=0.5, batch_size=8, has_ground_truth=True, logger=logger
                )

            llm_ft_f1 = llm_ft_results["overall_metrics"]["overall_f1_pct"]
            llm_ft_conf = llm_ft_results["overall_metrics"]["overall_confidence_pct"]

            logger.info(f"GLiNER FT (LLM labels) Results: F1={llm_ft_f1:.1f}%, Confidence={llm_ft_conf:.1f}%")

            # Cleanup evaluation model
            del eval_model
            cleanup_memory()
        else:
            llm_ft_f1 = 0.0
            llm_ft_conf = 0.0
            logger.error(f"No valid LLM labeled data for {n_examples} examples")

        # ===============================================================================
        # Training Phase 2: GLiNER FT on GT Labels
        # ===============================================================================

        logger.info(f"\nTraining GLiNER on GT labels ({n_examples} examples)")

        # Use ground truth labels from train_subset
        gt_labeled_data = []
        for example in train_subset:
            gt_labeled_data.append({
                "tokenized_text": example["tokenized_text"],
                "ner": example["ner"]  # Use ground truth labels
            })

        # Define adapter save path
        gt_adapter_path = os.path.join(models_dir, f"confidence_gt_model_{n_examples}")

        # Initialize model with LoRA using GLONER
        model = GLONER.default(logger=logger)
        model.to(device)

        # Train the model on GT labels
        train_lora_model(
            model=model,
            train_data=gt_labeled_data,
            eval_data=test_data,  # Small eval subset to speed up training
            training_config=training_config,
            adapter_save_path=gt_adapter_path,
            logger=logger
        )

        # Cleanup training model
        del model
        cleanup_memory()

        # ===============================================================================
        # Evaluation Phase 2: GLiNER FT on GT Labels
        # ===============================================================================

        logger.info(f"Evaluating GLiNER FT (GT labels) on FULL test set...")

        # Load model with trained adapter using GLONER
        eval_model = GLONER.load_with_adapter(gt_adapter_path, logger=logger)

        # Enhanced evaluation on FULL test set
        import torch
        with torch.no_grad():
            gt_ft_results = enhanced_evaluate(
                eval_model, test_data, entity_types,
                threshold=0.5, batch_size=8, has_ground_truth=True, logger=logger
            )

        gt_ft_f1 = gt_ft_results["overall_metrics"]["overall_f1_pct"]
        gt_ft_conf = gt_ft_results["overall_metrics"]["overall_confidence_pct"]

        logger.info(f"GLiNER FT (GT labels) Results: F1={gt_ft_f1:.1f}%, Confidence={gt_ft_conf:.1f}%")

        # Cleanup evaluation model
        del eval_model
        cleanup_memory()

        # ===============================================================================
        # Store Results
        # ===============================================================================

        results['no_worst_examples'].append(n_examples)
        results['gliner_ft_llm_f1'].append(llm_ft_f1)
        results['gliner_ft_gt_f1'].append(gt_ft_f1)
        results['confidence'].append((llm_ft_conf + gt_ft_conf) / 2)  # Average confidence
        results['avg_entities'].append(avg_entities)
        results['avg_input_tokens'].append(avg_input_tokens)
        results['avg_output_tokens'].append(avg_output_tokens)

        logger.info(f"Results stored for {n_examples} examples")

    # ===============================================================================
    # Results Analysis and Visualization
    # ===============================================================================

    logger.info(f"\nCreating Results DataFrame...")

    final_results_df = pd.DataFrame(results)

    # Configure pandas for full display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    logger.info("\n" + "="*60)
    logger.info("FINE-TUNING PERFORMANCE ANALYSIS RESULTS")
    logger.info("="*60)
    logger.info(final_results_df.to_string(index=False))

    # Reset pandas display options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '../results', LLM_BACKEND)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"confidence_finetuning_performance_{LLM_MODEL.replace(':', '_')}.csv")
    final_results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")

    # ===============================================================================
    # Visualization
    # ===============================================================================

    logger.info(f"\nGenerating Fine-tuning Trend Plot...")

    # Set style
    plt.style.use('default')
    sns.set_palette("viridis")

    # Create trend line plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot GLiNER FT on LLM labels
    ax.plot(
        final_results_df['no_worst_examples'], final_results_df['gliner_ft_llm_f1'],
        marker='o', markersize=8, linewidth=3,
        label='GLiNER FT (LLM Labels)', color='green', alpha=0.8
    )

    # Plot GLiNER FT on GT labels
    ax.plot(
        final_results_df['no_worst_examples'], final_results_df['gliner_ft_gt_f1'],
        marker='s', markersize=8, linewidth=3,
        label='GLiNER FT (GT Labels)', color='orange', alpha=0.8
    )

    # Formatting
    ax.set_title('Fine-tuning Performance: LLM vs GT Labels (Evaluated on Full Test Set)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Worst Confidence Examples (Training)', fontsize=14)
    ax.set_ylabel('F1 Score (%) on Full Test Set', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Add value annotations
    for i, (x, y1, y2) in enumerate(zip(final_results_df['no_worst_examples'],
                                       final_results_df['gliner_ft_llm_f1'],
                                       final_results_df['gliner_ft_gt_f1'])):
        ax.annotate(f'{y1:.1f}%', (x, y1), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=10, color='green')
        ax.annotate(f'{y2:.1f}%', (x, y2), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=10, color='orange')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(results_dir, f"confidence_finetuning_performance_trend_{LLM_MODEL.replace(':', '_')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {plot_path}")
    plt.show()

    # ===============================================================================
    # Final Summary
    # ===============================================================================

    logger.info(f"\nFine-tuning Analysis completed successfully!")
    logger.info(f"Best LLM FT F1: {max(results['gliner_ft_llm_f1']):.1f}% on {results['no_worst_examples'][results['gliner_ft_llm_f1'].index(max(results['gliner_ft_llm_f1']))]} examples")
    logger.info(f"Best GT FT F1: {max(results['gliner_ft_gt_f1']):.1f}% on {results['no_worst_examples'][results['gliner_ft_gt_f1'].index(max(results['gliner_ft_gt_f1']))]} examples")


if __name__ == "__main__":
    main()
