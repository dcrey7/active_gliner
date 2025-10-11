#!/usr/bin/env python3
"""
Confidence-driven fine-tuning experiment using the updated training pipeline.

Workflow per subset size:
1. Generate LLM labels with the unified LLMInference (training mode, Ollama Gemma3).
2. Fine-tune GLONER (GLiNER + LoRA) on the generated labels.
3. Fine-tune GLONER on ground-truth labels for comparison.
4. Evaluate both adapters with the new evaluate_gloner() utility.

All logging, caching, and evaluation use the latest abstractions.
"""

import gc
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
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
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
sys.path.append('../src')
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_PATH = (SCRIPT_DIR / "../src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))
PROJECT_ROOT = SRC_PATH.parent

# ---------------------------------------------------------------------------
# New pipeline imports
# ---------------------------------------------------------------------------
from config.settings import Settings  
from config.training_config import TRAINING_CONFIG  
from utils.logging import setup_logging, get_logger  
from utils.reproducibility import set_all_seeds  
from utils.device import setup_device, log_cuda_info  
from data.loader import load_mit_dataset  
from data.loader import load_json_file  
from generation.llm_inference import create_llm_train_labels  
from models.gloner import GLONER  
from training.trainer import train_lora_model  
from evaluation.eval import evaluate_gloner  

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOGGER_NAME = "ConfidenceFineTuning"
SUBSET_SIZES = [10,50,100,250,500,750,1000,1250,1500,1750,2000,2250,2500]
LLM_BACKEND = "ollama"
LLM_MODEL = "gemma3:12b"  # Ollama Gemma3




def ensure_directory(path: Path) -> Path:
    """Create directory (and parents) if missing."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_adapter(
    adapter_dir: Path,
    test_data: List[Dict[str, Any]],
    entity_types: List[str],
    device: torch.device,
    logger
) -> Dict[str, Any]:
    """Load adapter into GLONER, run predictions, and evaluate."""
    gloner = GLONER.for_inference(base_model_path=None, adapter_path=str(adapter_dir), logger=logger)
    gloner.to(device)
    device_str = str(device)

    predictions = gloner.predict(
        data=test_data,
        entity_types=entity_types,
        threshold=0.5,
        batch_size=8,
        device=device_str,
        flat_ner=True,
    )

    results = evaluate_gloner(predictions, test_data, entity_types)

    # Cleanup
    del gloner
    torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    settings = Settings()
    settings.setup()

    logger = setup_logging(log_dir=str(settings.logs_dir), logger_name=LOGGER_NAME)
    set_all_seeds(seed=settings.global_seed, logger=logger)
    device = setup_device(logger=logger)
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    log_cuda_info(logger)

    logger.info("=" * 80)
    logger.info("Confidence Fine-tuning Test (new pipeline)")
    logger.info("=" * 80)

    # -----------------------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------------------
    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file
    low_conf_path = PROJECT_ROOT / "results" / "high_mse_2500_examples.json"

    if not (test_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Test data or labels file not found.")
    if not low_conf_path.exists():
        raise FileNotFoundError(f"Low-confidence file missing: {low_conf_path}")

    test_data, entity_types = load_mit_dataset(str(test_data_path), str(labels_path), "test")
    low_conf_examples = load_json_file(str(low_conf_path))

    logger.info("Loaded %d test examples and %d entity types", len(test_data), len(entity_types))
    logger.info("Loaded %d low-confidence examples", len(low_conf_examples))

    # -----------------------------------------------------------------------
    # Initialize LLM generator (training mode with disk cache)
    # -----------------------------------------------------------------------
    llm_labeler = create_llm_train_labels(
        backend_type=LLM_BACKEND,
        model_name=LLM_MODEL,
        entity_types=entity_types,
        cache_type="disk",
        use_structured_output=False,
        logger=get_logger(LOGGER_NAME),
    )

    cache = getattr(llm_labeler, "cache", None)
    if cache:
        logger.info("Label cache directory: %s", cache.cache_dir)
        logger.info("Cached labels available: %d", cache.size())

    adapters_base = ensure_directory(settings.models_dir / "confidence_ft_adapters")
    results_rows: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Main experiment loop
    # -----------------------------------------------------------------------
    for n_examples in SUBSET_SIZES:
        logger.info("-" * 80)
        logger.info("Processing subset size: %d", n_examples)

        # Generate labels (will reuse disk cache across iterations)
        llm_result = llm_labeler.generate(
            examples=low_conf_examples,
            entity_types=entity_types,
            num_samples=n_examples,
            verbose=True,
        )

        llm_labeled_data = llm_result["all_labels"]
        logger.info("LLM labels obtained: %d examples", len(llm_labeled_data))
        logger.info("Token usage (input/out): %d / %d",
                    llm_result["total_input_tokens"],
                    llm_result["total_output_tokens"])

        if cache:
            logger.info("Cache size after generation: %d", cache.size())

        if not llm_labeled_data:
            logger.warning("No labeled data produced for %d examples, skipping.", n_examples)
            continue

        # Prepare ground-truth subset
        gt_subset = [
            {"tokenized_text": ex["tokenized_text"], "ner": ex["ner"]}
            for ex in low_conf_examples[:n_examples]
        ]

        # -------------------------------------------------------------------
        # Train & evaluate on LLM labels
        # -------------------------------------------------------------------
        llm_adapter_dir = ensure_directory(adapters_base / f"llm_{n_examples}")
        logger.info("Training GLONER on LLM labels (%d examples)...", len(llm_labeled_data))

        gloner_train = GLONER.for_training(logger=logger)
        gloner_train.to(device)

        train_lora_model(
            model=gloner_train.model,
            train_data=llm_labeled_data,
            eval_data=test_data,
            training_config=TRAINING_CONFIG,
            adapter_save_path=str(llm_adapter_dir),
            logger=logger,
        )

        # Release training model
        del gloner_train
        torch.cuda.empty_cache()
        gc.collect()

        llm_eval_results = evaluate_adapter(llm_adapter_dir, test_data, entity_types, device, logger)
        llm_overall = llm_eval_results["overall_metrics"]
        llm_f1 = llm_overall["overall_f1_pct"]
        logger.info("LLM labels adapter F1: %.2f%%", llm_f1)

        # -------------------------------------------------------------------
        # Train & evaluate on ground-truth labels
        # -------------------------------------------------------------------
        gt_adapter_dir = ensure_directory(adapters_base / f"gt_{n_examples}")
        logger.info("Training GLONER on ground-truth labels (%d examples)...", len(gt_subset))

        gloner_train_gt = GLONER.for_training(logger=logger)
        gloner_train_gt.to(device)

        train_lora_model(
            model=gloner_train_gt.model,
            train_data=gt_subset,
            eval_data=test_data,
            training_config=TRAINING_CONFIG,
            adapter_save_path=str(gt_adapter_dir),
            logger=logger,
        )

        del gloner_train_gt
        torch.cuda.empty_cache()
        gc.collect()

        gt_eval_results = evaluate_adapter(gt_adapter_dir, test_data, entity_types, device, logger)
        gt_overall = gt_eval_results["overall_metrics"]
        gt_f1 = gt_overall["overall_f1_pct"]
        logger.info("Ground-truth adapter F1: %.2f%%", gt_f1)

        results_rows.append({
            "num_examples": n_examples,
            "llm_overall_f1": llm_f1,
            "llm_example_accuracy": llm_overall["example_level_accuracy_pct"],
            "gt_overall_f1": gt_f1,
            "gt_example_accuracy": gt_overall["example_level_accuracy_pct"],
            "avg_entities_per_example": sum(len(ex["ner"]) for ex in llm_labeled_data) / len(llm_labeled_data),
            "llm_input_tokens": llm_result["total_input_tokens"],
            "llm_output_tokens": llm_result["total_output_tokens"],
        })

    # -----------------------------------------------------------------------
    # Summarize results
    # -----------------------------------------------------------------------
    if results_rows:
        results_df = pd.DataFrame(results_rows)
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            logger.info("Final results table:\n%s", results_df.to_string(index=False))

        results_dir = ensure_directory(PROJECT_ROOT / "results" / "confidence_ft")
        output_path = results_dir / "confidence_ft_summary.csv"
        results_df.to_csv(output_path, index=False)
        logger.info("Results saved to %s", output_path)

        # ===============================================================================
        # Visualization
        # ===============================================================================

        logger.info("\nGenerating performance comparison plot...")

        plt.style.use('default')
        sns.set_palette("husl")

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot GLiNER Base
        ax.plot(
            results_df['num_examples'],
            results_df['gt_overall_f1'],
            marker='o', markersize=10, linewidth=3,
            label='GLiNER GT Finetuned Model', color='blue', alpha=0.8
        )

        # Plot LLM
        ax.plot(
            results_df['num_examples'],
            results_df['llm_overall_f1'],
            marker='s', markersize=10, linewidth=3,
            label=f'GLiNER LLM Finetuned Model ({LLM_MODEL})', color='green', alpha=0.8
        )

        # Formatting
        ax.set_title(
            'GLINER Finetuning performance: GT vs LLM labels',
            fontsize=16, fontweight='bold', pad=20
        )
        ax.set_xlabel('Number of Worst Confidence Examples', fontsize=14)
        ax.set_ylabel('F1 Score (%)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='best')

        # Add value annotations
        for i, (x, y1, y2) in enumerate(zip(
            results_df['num_examples'],
            results_df['gt_overall_f1'],
            results_df['llm_overall_f1']
        )):
            ax.annotate(f'{y1:.1f}%', (x, y1), textcoords="offset points",
                       xytext=(0,10), ha='center', fontsize=10, color='blue')
            ax.annotate(f'{y2:.1f}%', (x, y2), textcoords="offset points",
                       xytext=(0,-15), ha='center', fontsize=10, color='green')

        plt.tight_layout()

        # Save plot
        plot_file = results_dir / f"confidence_finetuning_performance_trend_{LLM_MODEL.replace(':', '_')}.png"
        plt.savefig(str(plot_file), dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {plot_file}")
        plt.close()
    else:
        logger.warning("No experiments completed successfully.")





    logger.info("Confidence fine-tuning test finished.")




if __name__ == "__main__":
    main()
