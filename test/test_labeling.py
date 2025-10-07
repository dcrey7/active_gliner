#!/usr/bin/env python3
"""
Label generation smoke test using the new LLMInference training pipeline.

This script pulls a small subset of the precomputed low-confidence MIT-movie
examples and regenerates labels using the Cerebras structured API with the
Qwen-3 instruct model (non-thinking variant). Results are cached on disk so
reruns reuse prior generations.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import warnings
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

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Project imports (ensure src/ is on path)
# -----------------------------------------------------------------------------
sys.path.append('../src')

SCRIPT_DIR = Path(__file__).resolve().parent

from config.settings import Settings  # noqa: E402
from data.loader import load_json_file  # noqa: E402
from generation.llm_inference import create_llm_train_labels  # noqa: E402
from utils.logging import setup_logging, get_logger  # noqa: E402
from utils.reproducibility import set_all_seeds  # noqa: E402

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
LOGGER_NAME = "LLMLabelGenerationTest"
NUM_SAMPLES = 5
ENTITY_TYPES: List[str] = [
    "genre",
    "year",
    "plot",
    "average ratings",
    "actor",
    "title",
    "song",
    "character",
    "rating",
    "review",
    "director",
    "trailer",
]





def main():
    settings = Settings()
    settings.setup()

    logger = setup_logging(log_dir=str(settings.logs_dir), logger_name=LOGGER_NAME)
    set_all_seeds(seed=settings.global_seed, logger=logger)

    os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_device

    logger.info("=" * 80)
    logger.info("LLM LABEL GENERATION TEST (training mode, %d examples)", NUM_SAMPLES)
    logger.info("=" * 80)
    logger.info("Backend: cerebras (structured)")
    logger.info("Model: qwen-3-235b-a22b-instruct-2507")
    logger.info("Entity types: %s", ENTITY_TYPES)

    low_conf_path = (SCRIPT_DIR / "../results/high_mse_2500_examples.json").resolve()
    low_conf_examples = load_json_file(str(low_conf_path))
    logger.info("Loaded %d low-confidence examples from %s", len(low_conf_examples), low_conf_path)

    label_generator = create_llm_train_labels(
        backend_type="cerebras",
        model_name="qwen-3-235b-a22b-instruct-2507",
        entity_types=ENTITY_TYPES,
        cache_type="disk",
        use_structured_output=True,
        logger=get_logger(LOGGER_NAME),
    )

    cache = getattr(label_generator, "cache", None)
    if cache:
        logger.info("Cache directory: %s", cache.cache_dir)
        logger.info("Cached label count before run: %d", cache.size())

    logger.info("Generating up to %d labels using new training inference pipeline", NUM_SAMPLES)
    results = label_generator.generate(
        examples=low_conf_examples,
        entity_types=ENTITY_TYPES,
        num_samples=NUM_SAMPLES,
        verbose=True,
    )

    labels = results["all_labels"]
    logger.info("Label generation complete")
    logger.info("Total labels returned: %d", len(labels))
    logger.info("Input tokens consumed: %d", results["total_input_tokens"])
    logger.info("Output tokens produced: %d", results["total_output_tokens"])

    if cache:
        logger.info("Cached label count after run: %d", cache.size())

    # Log a quick preview of generated labels
    preview_count = min(3, len(labels))
    for idx in range(preview_count):
        example = labels[idx]
        logger.info("Example %d text: %s", idx + 1, " ".join(example["tokenized_text"][:20]))
        logger.info("Example %d entities: %s", idx + 1, example["ner"])

    logger.info("LLM label generation test completed successfully")


if __name__ == "__main__":
    main()
