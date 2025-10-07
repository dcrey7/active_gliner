import os
import sys
from pathlib import Path

import pandas as pd
import torch

# Ensure src/ is on PYTHONPATH
import sys
sys.path.append('../src')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from config.settings import Settings  # noqa: E402
from data.loader import load_mit_dataset  # noqa: E402
from evaluation.eval import evaluate_llm  # noqa: E402
from generation.llm_inference import create_llm_eval_labels  # noqa: E402
from utils.logging import get_logger, setup_logging  # noqa: E402
from utils.reproducibility import set_all_seeds  # noqa: E402


NUM_SAMPLES = 2442
LOGGER_NAME = "LLMEvaluationTest"


def log_device_info(logger):
    """Log CUDA device information."""
    logger.info("CUDA version: %s", torch.version.cuda)
    logger.info("Visible GPUs: %d", torch.cuda.device_count())

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        total_memory_gb = torch.cuda.get_device_properties(current_device).total_memory / 1024**3

        logger.info("Using device: cuda:%d", current_device)
        logger.info("GPU name: %s", device_name)
        logger.info("GPU memory: %.1f GB", total_memory_gb)
    else:
        logger.info("CUDA not available, falling back to CPU")


def main():
    settings = Settings()
    settings.setup()

    logger = setup_logging(log_dir=str(settings.logs_dir), logger_name=LOGGER_NAME)
    set_all_seeds(seed=settings.global_seed, logger=logger)
    log_device_info(logger)

    logger.info("=" * 80)
    logger.info("LLM EVALUATION TEST (subset of %d examples)", NUM_SAMPLES)
    logger.info("=" * 80)

    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file

    if not (test_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Test data or labels file not found! "
                                f"Expected {test_data_path} and {labels_path}")

    logger.info("Loading MIT test data from %s", test_data_path)
    test_data, entity_types = load_mit_dataset(str(test_data_path), str(labels_path), "test")
    logger.info("Loaded %d total test examples", len(test_data))
    logger.info("Entity types: %s", entity_types)

    logger.info("Preparing LLM predictor (backend=ollama, model=gemma3:12b)")
    llm_predictor = create_llm_eval_labels(
        backend_type="ollama",
        model_name="gemma3:12b",
        entity_types=entity_types,
        cache_type="disk",
        use_structured_output=False,
        logger=get_logger(LOGGER_NAME)
    )

    cache = getattr(llm_predictor, "cache", None)
    if cache:
        logger.info("Cache directory: %s", cache.cache_dir)
        logger.info("Existing cached predictions: %d", cache.size())

    logger.info("Generating LLM predictions for first %d examples", NUM_SAMPLES)
    llm_results = llm_predictor.generate(
        examples=test_data,
        entity_types=entity_types,
        num_samples=NUM_SAMPLES,
        verbose=True
    )

    logger.info("Generation complete")
    logger.info("Total predictions returned: %d", len(llm_results["all_labels"]))
    logger.info("Input tokens consumed: %d", llm_results["total_input_tokens"])
    logger.info("Output tokens produced: %d", llm_results["total_output_tokens"])

    if cache:
        cache.save_to_disk(reason="evaluation_run_completed")
        logger.info("Cache saved successfully")

    logger.info("Evaluating predictions against ground truth")
    eval_results = evaluate_llm(
        predictions=llm_results["all_labels"],
        data=test_data[:NUM_SAMPLES],
        entity_types=entity_types
    )

    overall = eval_results["overall_metrics"]
    logger.info("Overall F1: %.2f%%", overall["overall_f1_pct"])
    logger.info("Total examples: %d", overall["total_examples"])
    logger.info("Correct examples: %d", overall["correct_examples"])
    logger.info("Incorrect examples: %d", overall["incorrect_examples"])
    logger.info("Example accuracy: %.2f%%", overall["example_level_accuracy_pct"])
    logger.info("Entity accuracy: %.2f%%", overall["entity_level_accuracy_pct"])

    classification_df = eval_results["classification_report_df"]
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None
    ):
        table_str = classification_df.to_string(index=False)
    logger.info("Entity-level metrics table:\n%s", table_str)

    logger.info("LLM evaluation test complete")


if __name__ == "__main__":
    main()
