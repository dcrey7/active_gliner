#!/usr/bin/env python3
"""
Mixed Ratio Fine-tuning Experiment
Tests GLiNER fine-tuned on different GT/LLM label ratios
Evaluates fine-tuned models on FULL MIT test set

Uses new abstractions:
- GLONER for model initialization and loading
- create_label_generator for LLM labeling with disk caching
- train_lora_model for training
- NERValidator for validation
"""

import gc
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm import tqdm
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
from utils.logging import setup_logging, get_logger  
from config.training_config import TRAINING_CONFIG  
from utils.reproducibility import set_all_seeds  
from utils.device import setup_device, log_cuda_info  
from data.loader import load_mit_dataset  
from data.loader import load_json_file  
from generation.llm_inference import create_llm_train_labels  
from models.gloner import GLONER  
from training.trainer import train_lora_model  
from evaluation.eval import evaluate_gloner  
from data.transforms import create_mixed_training_data
from utils.memory import cleanup_memory


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOGGER_NAME = "MixedFineTuning"
SUBSET_SIZES = [10,50,100,250,500,750,1000,1250,1500,1750,2000,2250,2500]
GT_RATIOS = [0, 25, 50, 75, 100]
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
    """Mixed Ratio Fine-tuning Analysis"""

    # ===============================================================================
    # Setup and Configuration
    # ===============================================================================

    LOGGER_NAME = "MixedFintuning"

    settings = Settings()
    settings.setup()

    logger = setup_logging(log_dir=str(settings.logs_dir), logger_name=LOGGER_NAME)
    set_all_seeds(seed=settings.global_seed, logger=logger)
    device = setup_device(logger=logger)
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    log_cuda_info(logger)

    logger.info("=" * 80)
    logger.info("Mixed Fine-tuning Test (new pipeline)")
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




    # -----------------------------------------------------------------------
    # Main experiment loop
    # -----------------------------------------------------------------------
    # Results storage - single row per subset size
    results = {
        'no_worst_examples': [],
        'gliner_ft_0gt_100llm_f1': [],    # 0% GT, 100% LLM
        'gliner_ft_25gt_75llm_f1': [],    # 25% GT, 75% LLM
        'gliner_ft_50gt_50llm_f1': [],    # 50% GT, 50% LLM
        'gliner_ft_75gt_25llm_f1': [],    # 75% GT, 25% LLM
        'gliner_ft_100gt_0llm_f1': [],    # 100% GT, 0% LLM
        'confidence': [],
        
    }

    total_iterations = len(SUBSET_SIZES) * len(GT_RATIOS)
    logger.info(f"\nMixed Ratio Experiment Overview:")
    logger.info(f"   Subset sizes to test: {SUBSET_SIZES}")
    logger.info(f"   GT ratios to test: {GT_RATIOS}%")
    logger.info(f"   Total model trainings: {total_iterations}")
    logger.info(f"   Evaluation dataset: FULL test set ({len(test_data)} examples)")

    logger.info(f"\nStarting Mixed Ratio Analysis...")
    logger.info("-" * 60)

    for n_examples in tqdm(SUBSET_SIZES, desc="Training Mixed Ratios", position=0):
        logger.info(f"\nProcessing {n_examples} examples with 5 different ratios")

        # Get subset for training
        train_subset = low_conf_examples[:n_examples]

      
         # Generate labels (will reuse disk cache across iterations)
        logger.info(f"Generating LLM labels for {n_examples} examples (with disk caching)...")

        llm_gen_results = llm_labeler.generate(
            examples=low_conf_examples,
            entity_types=entity_types,
            num_samples=n_examples,
            verbose=True,
        )

        llm_labeled_data = llm_gen_results['all_labels']

        
        if cache:
            logger.info("Cache size after generation: %d", cache.size())

        if not llm_labeled_data:
            logger.warning("No labeled data produced for %d examples, skipping.", n_examples)
            continue


        logger.info(f"Generated: {len(llm_labeled_data)} examples")

        # ===============================================================================
        # Train 5 Models with Different Ratios
        # ===============================================================================

        ratio_f1_scores = []
        avg_confidence = 0.0

        for gt_ratio in GT_RATIOS:
            logger.info(f"\nTraining GLiNER with {gt_ratio}% GT + {100-gt_ratio}% LLM labels")

            # Create mixed training data   
            adapters_base = ensure_directory(settings.models_dir / "confidence_mixed_adapters")
            results_rows: List[Dict[str, Any]] = []

            if gt_ratio == 0:
                # Pure LLM labels
                mixed_training_data = llm_labeled_data
                logger.info(f"   Using 100% LLM labels ({len(mixed_training_data)} examples)")
            elif gt_ratio == 100:
                # Pure GT labels
                mixed_training_data = [{
                    "tokenized_text": ex["tokenized_text"],
                    "ner": ex["ner"]
                } for ex in train_subset]
                logger.info(f"   Using 100% GT labels ({len(mixed_training_data)} examples)")
            else:
                # Mixed labels
                mixed_training_data = create_mixed_training_data(
                    train_subset, llm_labeled_data, gt_ratio
                )
                n_gt = int(len(mixed_training_data) * gt_ratio / 100)
                n_llm = len(mixed_training_data) - n_gt
                logger.info(f"   Using {n_gt} GT + {n_llm} LLM labels ({len(mixed_training_data)} total)")

            # Define adapter save path
            models_dir = os.path.join(os.path.dirname(__file__), '../models')
            adapter_path = os.path.join(models_dir, f"mixed_ratio_model_{n_examples}_{gt_ratio}gt")
            os.makedirs(models_dir, exist_ok=True)

            # Initialize model with LoRA using GLONER
            gloner = GLONER.for_training(logger=logger)
            gloner.to(device)

            # Train the model
            train_lora_model(
                model=gloner.model,
                train_data=mixed_training_data,
                eval_data=test_data,  # Small eval subset for speed
                training_config=TRAINING_CONFIG,
                adapter_save_path=adapter_path,
                logger=logger
            )

            # Cleanup training model
            del gloner
            cleanup_memory()

            # ===============================================================================
            # Evaluation
            # ===============================================================================

            logger.info(f"Evaluating {gt_ratio}% GT model on FULL test set...")

            # Enhanced evaluation on FULL test set
            eval_results = evaluate_adapter(adapter_path, test_data, entity_types, device, logger)



            ratio_f1 = eval_results["overall_metrics"]["overall_f1_pct"]
            ratio_conf = eval_results["overall_metrics"]["overall_confidence_pct"]

            logger.info(f"{gt_ratio}% GT Results: F1={ratio_f1:.1f}%, Confidence={ratio_conf:.1f}%")

            ratio_f1_scores.append(ratio_f1)
            avg_confidence += ratio_conf

            # Cleanup
            cleanup_memory()

        # ===============================================================================
        # Store Results for This Subset Size
        # ===============================================================================

        results['no_worst_examples'].append(n_examples)
        results['gliner_ft_0gt_100llm_f1'].append(ratio_f1_scores[0])   # 0% GT
        results['gliner_ft_25gt_75llm_f1'].append(ratio_f1_scores[1])   # 25% GT
        results['gliner_ft_50gt_50llm_f1'].append(ratio_f1_scores[2])   # 50% GT
        results['gliner_ft_75gt_25llm_f1'].append(ratio_f1_scores[3])   # 75% GT
        results['gliner_ft_100gt_0llm_f1'].append(ratio_f1_scores[4])   # 100% GT
        results['confidence'].append(avg_confidence / len(GT_RATIOS))
        

        logger.info(f"Results stored for {n_examples} examples")
        logger.info(f"F1 Scores: 0%GT={ratio_f1_scores[0]:.1f}%, 25%GT={ratio_f1_scores[1]:.1f}%, "
                   f"50%GT={ratio_f1_scores[2]:.1f}%, 75%GT={ratio_f1_scores[3]:.1f}%, "
                   f"100%GT={ratio_f1_scores[4]:.1f}%")

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
    logger.info("MIXED RATIO FINE-TUNING ANALYSIS RESULTS")
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
    results_path = os.path.join(results_dir, f"mixed_ratio_finetuning_performance_{LLM_MODEL.replace(':', '_')}.csv")
    final_results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")

    # ===============================================================================
    # Visualization
    # ===============================================================================

    logger.info(f"\nGenerating Mixed Ratio Performance Plot...")

    # Set style
    plt.style.use('default')
    sns.set_palette("viridis")

    # Create trend line plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Plot all 5 ratio curves
    ratio_columns = [
        ('gliner_ft_0gt_100llm_f1', '0% GT + 100% LLM', 'red'),
        ('gliner_ft_25gt_75llm_f1', '25% GT + 75% LLM', 'orange'),
        ('gliner_ft_50gt_50llm_f1', '50% GT + 50% LLM', 'blue'),
        ('gliner_ft_75gt_25llm_f1', '75% GT + 25% LLM', 'lightgreen'),
        ('gliner_ft_100gt_0llm_f1', '100% GT + 0% LLM', 'green')
    ]

    for col_name, label, color in ratio_columns:
        ax.plot(
            final_results_df['no_worst_examples'], final_results_df[col_name],
            marker='o', markersize=8, linewidth=3,
            label=label, color=color, alpha=0.8
        )

    # Formatting
    ax.set_title('Mixed Ratio Fine-tuning Performance: GT vs LLM Labels (Evaluated on Full Test Set)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Worst Confidence Examples (Training)', fontsize=14)
    ax.set_ylabel('F1 Score (%) on Full Test Set', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')

    # Add value annotations
    for i, (col_name, label, color) in enumerate(ratio_columns):
        if i % 2 == 0:  # Annotate every other line to avoid clutter
            for j, (x, y) in enumerate(zip(final_results_df['no_worst_examples'],
                                         final_results_df[col_name])):
                ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                           xytext=(0,15), ha='center', fontsize=9, color=color)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(results_dir, f"mixed_ratio_finetuning_performance_{LLM_MODEL.replace(':', '_')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {plot_path}")
    plt.show()

    # ===============================================================================
    # Final Summary
    # ===============================================================================

    logger.info(f"\nMixed Ratio Analysis completed successfully!")

    # Find best performance for each ratio
    for col_name, label, _ in ratio_columns:
        best_f1 = max(results[col_name])
        best_idx = results[col_name].index(best_f1)
        best_examples = results['no_worst_examples'][best_idx]
        logger.info(f"{label}: Best F1={best_f1:.1f}% with {best_examples} examples")

    logger.info(f"Total labels cached: {len(llm_labeler.cache.get_all())} examples")


if __name__ == "__main__":
    main()
