#!/usr/bin/env python3
"""
GLONER smoke test using the updated infrastructure.

This script:
1. Applies full settings/env configuration (including GPU visibility).
2. Loads the MIT movie test split.
3. Instantiates a trainable GLONER (GLiNER + LoRA) model.
4. Runs predictions and evaluates with the new evaluation pipeline.
"""

import sys
sys.path.append('../src')

from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
# Check CUDA
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs visible: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    device = "cuda"
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = "cpu"
    print("CUDA not available, using CPU")
from config.settings import Settings  # noqa: E402
from utils.logging import setup_logging  # noqa: E402
from utils.reproducibility import set_all_seeds  # noqa: E402
from utils.device import setup_device, log_cuda_info  # noqa: E402
from data.loader import load_mit_dataset  # noqa: E402
from models.gloner import GLONER  # noqa: E402
from evaluation.eval import evaluate_gloner  # noqa: E402


def main():
    settings = Settings()
    settings.setup()

    logger = setup_logging(log_dir=str(settings.logs_dir), logger_name="GLONERTest")
    set_all_seeds(seed=settings.global_seed, logger=logger)

    device = setup_device(logger=logger)
    log_cuda_info(logger)

    logger.info("Default GLiNER model: %s", settings.model_name)
    logger.info("Default max length: %d", settings.model_max_length)

    test_path = settings.data_path / settings.test_file
    label_path = settings.data_path / settings.labels_file
    test_data, entity_types = load_mit_dataset(str(test_path), str(label_path), "test")
    logger.info("Loaded %d test examples | Entity types: %s", len(test_data), entity_types)

    gloner = GLONER.for_training(logger=logger)
    gloner.to(device)

    predictions = gloner.predict(
        data=test_data,
        entity_types=entity_types,
        threshold=settings.prediction_threshold,
        batch_size=settings.batch_size,
        device=str(device),
        flat_ner=True,
    )
    logger.info("Generated predictions for %d examples", len(predictions))

    gliner_results = gloner.evaluate(test_data, entity_types, batch_size=8, threshold=0.5
    )
    logger.info("GLiNER evaluation complete")
    logger.info(f"gliner results : {gliner_results}")

    eval_results = evaluate_gloner(predictions, test_data, entity_types, has_ground_truth=True)
    metrics = eval_results["overall_metrics"]
    logger.info("GLONER evaluation complete")
    logger.info("Overall F1: %.2f%% | Example accuracy: %.2f%% | Entity accuracy: %.2f%%",
                metrics["overall_f1_pct"],
                metrics["example_level_accuracy_pct"],
                metrics["entity_level_accuracy_pct"])


if __name__ == "__main__":
    main()
