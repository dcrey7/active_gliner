#!/usr/bin/env python3
"""
Test script for Active GLiNER package API
Demonstrates the 5 main entry points: zeroshot, ranking, finetune, predict, evaluate

Tests:
- zeroshot(): Zero-shot prediction with GLiNER
- ranking(): Uncertainty-based example selection
- finetune(): Fine-tuning with ground truth
- predict(): Prediction with fine-tuned adapter
- evaluate(): Evaluation of predictions
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import gc
from pathlib import Path

import torch

# Add src to path
sys.path.append('../src')

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_PATH = (SCRIPT_DIR / "../src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))
PROJECT_ROOT = SRC_PATH.parent

# Import main API
from main import zeroshot, ranking, finetune, predict, evaluate
from config.settings import Settings
from utils.logging import setup_logging
from utils.reproducibility import set_all_seeds
from utils.device import setup_device, log_cuda_info
from data.loader import load_mit_dataset
from utils.memory import cleanup_memory

LOGGER_NAME = "PackageAPITest"


def main():
    """Test the Active GLiNER package API"""

    # Setup
    settings = Settings()
    settings.setup()

    logger = setup_logging(log_dir=str(settings.logs_dir), logger_name=LOGGER_NAME)
    set_all_seeds(seed=settings.global_seed, logger=logger)
    device = setup_device(logger=logger)
    log_cuda_info(logger)

    logger.info("=" * 80)
    logger.info("TESTING ACTIVE GLINER PACKAGE API")
    logger.info("=" * 80)

    # Load data
    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file

    test_data, entity_types = load_mit_dataset(
        str(test_data_path),
        str(labels_path),
        "test"
    )

    # Use small subset for testing
    test_subset = test_data[:50]

    logger.info(f"Loaded {len(test_subset)} test examples")
    logger.info(f"Entity types: {entity_types}")

    # ===================================================================
    # 1. ZEROSHOT - Zero-shot prediction
    # ===================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: ZEROSHOT PREDICTION")
    logger.info("=" * 80)

    predictions = zeroshot(
        data=test_subset,
        entity_types=entity_types,
        threshold=0.5,
        batch_size=8,
        device=str(device),
        logger=logger
    )

    logger.info(f"Generated {len(predictions)} predictions")

    # ===================================================================
    # 2. RANKING - Select uncertain examples
    # ===================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: RANKING BY UNCERTAINTY")
    logger.info("=" * 80)

    uncertain_examples = ranking(
        predictions=predictions,
        data=test_subset,
        entity_types=entity_types,
        n_examples=10,
        strategy='mse',
        logger=logger
    )

    logger.info(f"Selected {len(uncertain_examples)} most uncertain examples")

    # ===================================================================
    # 3. FINETUNE - Train with ground truth
    # ===================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: FINETUNE WITH GT LABELS")
    logger.info("=" * 80)

    # Use top 10 uncertain for training
    train_subset = uncertain_examples[:10]
    eval_subset = test_subset[10:20]

    adapter_path = str(settings.models_dir / "test_adapter_api")

    finetune(
        training_data=train_subset,
        eval_data=eval_subset,
        entity_types=entity_types,
        adapter_save_path=adapter_path,
        logger=logger
    )

    logger.info(f"Adapter saved to: {adapter_path}")

    # ===================================================================
    # 4. PREDICT - Use trained adapter
    # ===================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: PREDICT WITH TRAINED ADAPTER")
    logger.info("=" * 80)

    finetuned_predictions = predict(
        data=test_subset,
        entity_types=entity_types,
        adapter_path=adapter_path,
        threshold=0.5,
        batch_size=8,
        device=str(device),
        logger=logger
    )

    logger.info(f"Generated {len(finetuned_predictions)} predictions with adapter")

    # ===================================================================
    # 5. EVALUATE - Compare results
    # ===================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: EVALUATE MODELS")
    logger.info("=" * 80)

    # Evaluate zero-shot
    logger.info("\nEvaluating zero-shot predictions...")
    zeroshot_results = evaluate(
        data=test_subset,
        entity_types=entity_types,
        predictions=predictions,
        model_type='gloner',
        has_ground_truth=True,
        logger=logger
    )

    # Evaluate fine-tuned
    logger.info("\nEvaluating fine-tuned predictions...")
    finetuned_results = evaluate(
        data=test_subset,
        entity_types=entity_types,
        predictions=finetuned_predictions,
        model_type='gloner',
        has_ground_truth=True,
        logger=logger
    )

    # Compare
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 80)

    zs_f1 = zeroshot_results['overall_metrics']['overall_f1_pct']
    ft_f1 = finetuned_results['overall_metrics']['overall_f1_pct']

    logger.info(f"Zero-shot F1:    {zs_f1:.2f}%")
    logger.info(f"Fine-tuned F1:   {ft_f1:.2f}%")
    logger.info(f"Improvement:     {ft_f1 - zs_f1:.2f}%")

    logger.info("\n" + "=" * 80)
    logger.info("PACKAGE API TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
