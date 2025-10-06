#!/usr/bin/env python3
"""
Mixed Ratio Fine-tuning Experiment with Enhanced API Quota Handling
Tests GLiNER fine-tuned on different GT/LLM label ratios with graceful failure handling

Uses new abstractions:
- GLONER for model initialization and loading
- create_label_generator with Cerebras API and structured output
- Disk caching for persistence across runs
- NERValidator for validation
- Graceful handling of API quota limits
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')

# Add src path
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.append(src_path)

# Import with new abstractions
from config import Settings, GLOBAL_SEED
from utils import setup_logging, set_all_seeds, setup_device, cleanup_memory
from data import load_mit_dataset, NERValidator
from evaluation import enhanced_evaluate
from generation import create_label_generator
from training import train_lora_model
from models.gloner import GLONER


def create_mixed_training_data(examples, llm_labels, gt_ratio):
    """Create training data with specified GT/LLM ratio"""
    n_examples = len(examples)
    n_gt = int(n_examples * gt_ratio / 100)

    gt_indices = random.sample(range(n_examples), n_gt)

    mixed_data = []
    for i, (example, llm_example) in enumerate(zip(examples, llm_labels)):
        if i in gt_indices:
            mixed_data.append({
                "tokenized_text": example["tokenized_text"],
                "ner": example["ner"]
            })
        else:
            mixed_data.append({
                "tokenized_text": llm_example["tokenized_text"],
                "ner": llm_example["ner"]
            })

    return mixed_data


def main():
    """Enhanced Mixed Ratio Fine-tuning Analysis with Quota Handling"""

    settings = Settings()
    settings.setup()
    logger = setup_logging(log_dir=str(settings.logs_dir))
    set_all_seeds(seed=GLOBAL_SEED, logger=logger)
    device = setup_device(logger=logger)

    # API Configuration - Cerebras with structured output
    LLM_BACKEND = "cerebras"
    LLM_MODEL = "llama3.1-8b"
    USE_STRUCTURED = True  # Use JSON schema validation

    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file

    if not (test_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Test data or labels file not found!")

    test_data, entity_types = load_mit_dataset(str(test_data_path), str(labels_path), "test")
    logger.info(f"Loaded FULL test data: {len(test_data)} examples, {len(entity_types)} entity types")

    logger.info("Loading pre-saved low confidence examples...")
    low_conf_file = os.path.join(os.path.dirname(__file__), '../results/high_mse_2500_examples.json')
    with open(low_conf_file, 'r') as f:
        low_n = json.load(f)
    logger.info(f"Loaded {len(low_n)} low confidence examples for training")

    # Initialize label generator with disk cache and structured output
    try:
        logger.info("Initializing Cerebras label generator with disk cache and structured output...")
        label_generator = create_label_generator(
            backend_type=LLM_BACKEND,
            model_name=LLM_MODEL,
            cache_type='disk',
            use_structured_output=USE_STRUCTURED
        )
        logger.info(f"Enhanced LLM Labeler: {LLM_BACKEND} - {LLM_MODEL}")
        logger.info("Structured Output: Enabled (JSON schema validation)")
    except Exception as e:
        logger.error(f"Failed to initialize label generator: {e}")
        logger.error("Please check CEREBRAS_API_KEY environment variable")
        return

    # Initialize validator
    validator = NERValidator(entity_types=entity_types, logger=logger)

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

    subset_sizes = [2, 4, 8, 10]  # Small sizes for testing
    gt_ratios = [0, 25, 50, 75, 100]

    results = {
        'no_worst_examples': [],
        'gliner_ft_0gt_100llm_f1': [],
        'gliner_ft_25gt_75llm_f1': [],
        'gliner_ft_50gt_50llm_f1': [],
        'gliner_ft_75gt_25llm_f1': [],
        'gliner_ft_100gt_0llm_f1': [],
        'confidence': [],
        'avg_entities': [],
        'avg_input_tokens': [],
        'avg_output_tokens': [],
        'completion_status': [],
        'completion_percentage': []
    }

    total_iterations = len(subset_sizes) * len(gt_ratios)
    logger.info(f"\nEnhanced Mixed Ratio Experiment Overview:")
    logger.info(f"   Subset sizes to test: {subset_sizes}")
    logger.info(f"   GT ratios to test: {gt_ratios}%")
    logger.info(f"   Total model trainings: {total_iterations}")
    logger.info(f"   Evaluation dataset: FULL test set ({len(test_data)} examples)")
    logger.info(f"   API resilience: Enabled")
    logger.info(f"   Incremental saving: Enabled")

    logger.info(f"\nStarting Enhanced Mixed Ratio Analysis...")
    logger.info("-" * 60)

    experiment_interrupted = False

    for subset_idx, n_examples in enumerate(tqdm(subset_sizes, desc="Training Mixed Ratios", position=0)):
        if experiment_interrupted:
            logger.warning(f"Skipping remaining subset sizes due to quota limit")
            break

        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {subset_idx+1}/{len(subset_sizes)}: Processing {n_examples} examples")
        logger.info(f"{'='*60}")

        train_subset = low_n[:n_examples]

        logger.info(f"Generating LLM labels for {n_examples} examples...")

        try:
            # Generate labels with disk caching
            llm_gen_results = label_generator.generate(
                low_n_examples=train_subset,
                num_samples=n_examples,
                entity_types=entity_types
            )

            llm_labeled_data = llm_gen_results['all_labels']
            actual_examples = n_examples
            completion_status = "complete"
            completion_pct = 100.0

            logger.info(f"Successfully generated {len(llm_labeled_data)} labels")

        except Exception as e:
            # Handle API quota or other errors gracefully
            logger.error("="*60)
            logger.error(f"ERROR DURING LABEL GENERATION: {str(e)}")
            logger.error("="*60)

            # Check if we have partial results from cache
            if hasattr(label_generator, 'cache') and len(label_generator.cache.get_all()) > 0:
                cached_labels = label_generator.cache.get_all()
                llm_labeled_data = cached_labels[:n_examples]
                actual_examples = len(llm_labeled_data)
                completion_status = "partial_from_cache"
                completion_pct = (actual_examples / n_examples) * 100
                logger.warning(f"Using {actual_examples} labels from cache ({completion_pct:.1f}% complete)")
                experiment_interrupted = True
            else:
                logger.error("No cached data available, skipping this iteration")
                continue

        # Validate LLM labels
        logger.info("Validating LLM labels...")
        llm_labeled_data, llm_report = validator.validate(llm_labeled_data, strict=True)
        logger.info(f"Validation: {len(llm_labeled_data)} valid examples")

        if len(llm_labeled_data) == 0:
            logger.error(f"No valid LLM labeled data for {n_examples} examples")
            logger.error("Skipping this iteration...")
            continue

        # Calculate metrics
        avg_entities = sum(len(ex['ner']) for ex in llm_labeled_data) / len(llm_labeled_data)
        avg_input_tokens = llm_gen_results['total_input_tokens'] / len(llm_labeled_data) if len(llm_labeled_data) > 0 else 0
        avg_output_tokens = llm_gen_results['total_output_tokens'] / len(llm_labeled_data) if len(llm_labeled_data) > 0 else 0

        logger.info(f"Metrics: avg_entities={avg_entities:.1f}, completion={completion_pct:.1f}%")
        logger.info(f"Token metrics: input={avg_input_tokens:.0f}, output={avg_output_tokens:.0f}")

        ratio_f1_scores = []
        avg_confidence = 0.0

        effective_examples = len(llm_labeled_data)

        for gt_ratio in gt_ratios:
            logger.info(f"\nTraining GLiNER with {gt_ratio}% GT + {100-gt_ratio}% LLM labels")
            logger.info(f"   Available data: {effective_examples} examples")

            if effective_examples < 1:
                logger.warning(f"   Insufficient data ({effective_examples} examples) - recording zero F1")
                ratio_f1_scores.append(0.0)
                continue

            if gt_ratio == 0:
                mixed_training_data = llm_labeled_data
                logger.info(f"   Using 100% LLM labels ({len(mixed_training_data)} examples)")
            elif gt_ratio == 100:
                mixed_training_data = [{
                    "tokenized_text": ex["tokenized_text"],
                    "ner": ex["ner"]
                } for ex in train_subset[:effective_examples]]
                logger.info(f"   Using 100% GT labels ({len(mixed_training_data)} examples)")
            else:
                gt_subset = train_subset[:effective_examples]
                mixed_training_data = create_mixed_training_data(
                    gt_subset, llm_labeled_data, gt_ratio
                )
                n_gt = int(len(mixed_training_data) * gt_ratio / 100)
                n_llm = len(mixed_training_data) - n_gt
                logger.info(f"   Using {n_gt} GT + {n_llm} LLM labels ({len(mixed_training_data)} total)")

            models_dir = os.path.join(os.path.dirname(__file__), '../models')
            adapter_path = os.path.join(models_dir, f"mixed_ratio_api_model_{effective_examples}_{gt_ratio}gt")
            os.makedirs(models_dir, exist_ok=True)

            try:
                model = GLONER.default(logger=logger)
                model.to(device)

                train_lora_model(
                    model=model,
                    train_data=mixed_training_data,
                    eval_data=test_data[:100],
                    training_config=training_config,
                    adapter_save_path=adapter_path,
                    logger=logger
                )

                del model
                cleanup_memory()

                logger.info(f"Evaluating {gt_ratio}% GT model on FULL test set...")

                eval_model = GLONER.load_with_adapter(adapter_path, logger=logger)

                import torch
                with torch.no_grad():
                    eval_results = enhanced_evaluate(
                        eval_model, test_data, entity_types,
                        threshold=0.5, batch_size=8, has_ground_truth=True, logger=logger
                    )

                ratio_f1 = eval_results["overall_metrics"]["overall_f1_pct"]
                ratio_conf = eval_results["overall_metrics"]["overall_confidence_pct"]

                logger.info(f"{gt_ratio}% GT Results: F1={ratio_f1:.1f}%, Confidence={ratio_conf:.1f}%")

                ratio_f1_scores.append(ratio_f1)
                avg_confidence += ratio_conf

                del eval_model
                cleanup_memory()

            except Exception as e:
                logger.error(f"Training/evaluation failed for {gt_ratio}% GT ratio: {str(e)}")
                ratio_f1_scores.append(0.0)

        while len(ratio_f1_scores) < 5:
            ratio_f1_scores.append(0.0)

        results['no_worst_examples'].append(actual_examples)
        results['gliner_ft_0gt_100llm_f1'].append(ratio_f1_scores[0])
        results['gliner_ft_25gt_75llm_f1'].append(ratio_f1_scores[1])
        results['gliner_ft_50gt_50llm_f1'].append(ratio_f1_scores[2])
        results['gliner_ft_75gt_25llm_f1'].append(ratio_f1_scores[3])
        results['gliner_ft_100gt_0llm_f1'].append(ratio_f1_scores[4])
        results['confidence'].append(avg_confidence / len(gt_ratios) if avg_confidence > 0 else 0.0)
        results['avg_entities'].append(avg_entities)
        results['avg_input_tokens'].append(avg_input_tokens)
        results['avg_output_tokens'].append(avg_output_tokens)
        results['completion_status'].append(completion_status)
        results['completion_percentage'].append(completion_pct)

        logger.info(f"Results stored for {actual_examples} examples")
        logger.info(f"F1 Scores: 0%GT={ratio_f1_scores[0]:.1f}%, 25%GT={ratio_f1_scores[1]:.1f}%, "
                   f"50%GT={ratio_f1_scores[2]:.1f}%, 75%GT={ratio_f1_scores[3]:.1f}%, "
                   f"100%GT={ratio_f1_scores[4]:.1f}%")
        logger.info(f"Status: {completion_status} ({completion_pct:.1f}% complete)")

        # Save incremental results
        temp_df = pd.DataFrame(results)
        results_dir = os.path.join(os.path.dirname(__file__), '../results/api')
        os.makedirs(results_dir, exist_ok=True)
        incremental_path = os.path.join(results_dir, "mixed_ratio_performance_incremental.csv")
        temp_df.to_csv(incremental_path, index=False)
        logger.info(f"Saved incremental results: {len(temp_df)} iterations completed")

        if experiment_interrupted:
            logger.error("="*60)
            logger.error("EXPERIMENT INTERRUPTED")
            logger.error("="*60)
            logger.error(f"Completed iterations: {len(results['no_worst_examples'])}")
            logger.error(f"Last successful size: {actual_examples} examples")
            logger.error("Proceeding to final analysis with collected data...")
            logger.error("="*60)
            break

    # Save final results and create visualization (similar to mixed_test_FT.py)
    logger.info(f"\nCreating Final Results DataFrame...")

    final_results_df = pd.DataFrame(results)
    final_results_df = final_results_df[final_results_df['no_worst_examples'] > 0]

    completed_iterations = len(final_results_df)
    planned_iterations = len(subset_sizes)

    logger.info("\n" + "="*60)
    logger.info("ENHANCED MIXED RATIO FINE-TUNING ANALYSIS RESULTS (API)")
    if completed_iterations < planned_iterations:
        logger.warning(f"PARTIAL RESULTS: {completed_iterations}/{planned_iterations} iterations")
    else:
        logger.info(f"COMPLETE RESULTS: {completed_iterations}/{planned_iterations} iterations")
    logger.info("="*60)

    # Configure pandas and display
    pd.set_option('display.max_columns', None)
    logger.info("\n" + final_results_df.to_string(index=False))
    pd.reset_option('display.max_columns')

    # Save results
    results_filename = f"mixed_ratio_api_performance_{LLM_MODEL}"
    if completed_iterations < planned_iterations:
        results_filename += "_partial"

    results_path = os.path.join(results_dir, f"{results_filename}.csv")
    final_results_df.to_csv(results_path, index=False)
    logger.info(f"\nFinal results saved to: {results_path}")

    logger.info(f"\nEnhanced Mixed Ratio Analysis Summary:")
    logger.info(f"Completed iterations: {completed_iterations}/{planned_iterations}")
    logger.info(f"Total labels cached: {len(label_generator.cache.get_all())} examples")


if __name__ == "__main__":
    main()
