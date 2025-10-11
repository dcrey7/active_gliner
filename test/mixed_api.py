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
import gc
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
from pathlib import Path
from typing import Dict, List, Any
import torch

warnings.filterwarnings('ignore')

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
LLM_BACKEND = "cerebras"
LLM_MODEL = "qwen-3-235b-a22b-instruct-2507" 


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
    """Enhanced Mixed Ratio Fine-tuning Analysis with Quota Handling"""

    # ===============================================================================
    # Setup and Configuration
    # ===============================================================================

    LOGGER_NAME = "MixedFintuning_API"

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
    label_generator = create_llm_train_labels(
        backend_type=LLM_BACKEND,
        model_name=LLM_MODEL,
        entity_types=entity_types,
        cache_type="disk",
        use_structured_output=False,
        logger=get_logger(LOGGER_NAME),
    )

    cache = getattr(label_generator, "cache", None)
    if cache:
        logger.info("Label cache directory: %s", cache.cache_dir)
        logger.info("Cached labels available: %d", cache.size())


    # -----------------------------------------------------------------------
    # Main experiment loop
    # -----------------------------------------------------------------------
    # Results storage - single row per subset size
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

    total_iterations = len(SUBSET_SIZES) * len(GT_RATIOS)
    logger.info(f"\nEnhanced Mixed Ratio Experiment Overview:")
    logger.info(f"   Subset sizes to test: {SUBSET_SIZES}")
    logger.info(f"   GT ratios to test: {GT_RATIOS}%")
    logger.info(f"   Total model trainings: {total_iterations}")
    logger.info(f"   Evaluation dataset: FULL test set ({len(test_data)} examples)")
    logger.info(f"   API resilience: Enabled")
    logger.info(f"   Incremental saving: Enabled")

    logger.info(f"\nStarting Enhanced Mixed Ratio Analysis...")
    logger.info("-" * 60)

    experiment_interrupted = False

    for subset_idx, n_examples in enumerate(tqdm(SUBSET_SIZES, desc="Training Mixed Ratios", position=0)):
        if experiment_interrupted:
            logger.warning(f"Skipping remaining subset sizes due to quota limit")
            break

        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {subset_idx+1}/{len(SUBSET_SIZES)}: Processing {n_examples} examples")
        logger.info(f"{'='*60}")

        train_subset = low_conf_examples[:n_examples]

        logger.info(f"Generating LLM labels for {n_examples} examples...")

        try:
            # Generate labels with disk caching
            llm_gen_results = label_generator.generate(
            examples=low_conf_examples,
            entity_types=entity_types,
            num_samples=n_examples,
            verbose=True,
            )

            llm_labeled_data = llm_gen_results['all_labels']
            actual_examples = len(llm_labeled_data)

            # Check if we got fewer labels than requested (quota may have been hit)
            if actual_examples < n_examples:
                completion_status = "partial_quota"
                completion_pct = (actual_examples / n_examples) * 100
                logger.warning(f"Got {actual_examples}/{n_examples} labels ({completion_pct:.1f}% complete)")
                logger.warning("Likely hit API quota - experiment will stop after this iteration")
                experiment_interrupted = True
            else:
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

        for gt_ratio in GT_RATIOS:
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
                gloner = GLONER.for_training(logger=logger)
                gloner.to(device)

                train_lora_model(
                    model=gloner.model,
                    train_data=mixed_training_data,
                    eval_data=test_data,
                    training_config=TRAINING_CONFIG,
                    adapter_save_path=adapter_path,
                    logger=logger
                )

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
        results['confidence'].append(avg_confidence / len(GT_RATIOS) if avg_confidence > 0 else 0.0)
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
        results_dir = os.path.join(os.path.dirname(__file__), f'../results/{LLM_BACKEND}/api')
        os.makedirs(results_dir, exist_ok=True)
        incremental_path = os.path.join(results_dir, "mixed_ratio_performance_incremental.csv")
        temp_df.to_csv(incremental_path, index=False)
        logger.info(f"Saved incremental results: {len(temp_df)} iterations completed")

        # Also save a snapshot as "latest" in case experiment crashes
        snapshot_path = os.path.join(results_dir, f"mixed_ratio_api_performance_{LLM_MODEL}_latest.csv")
        temp_df.to_csv(snapshot_path, index=False)
        logger.info(f"Saved snapshot: {snapshot_path}")

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
    planned_iterations = len(SUBSET_SIZES)

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


    plt.style.use('default')
    sns.set_palette("viridis")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
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
    
    for i, (idx, row) in enumerate(final_results_df.iterrows()):
        if row['completion_status'] != 'complete':
            for col_name, label, color in ratio_columns:
                y_val = row[col_name]
                if y_val > 0:
                    ax.scatter(row['no_worst_examples'], y_val, 
                             marker='x', s=100, color=color, alpha=0.7)
    
    title = 'Enhanced Mixed Ratio Fine-tuning Performance: GT vs LLM Labels'
    if completed_iterations < planned_iterations:
        title += f'\n(Partial Results - Stopped at {final_results_df["no_worst_examples"].iloc[-1]} examples)'
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Worst Confidence Examples (Training)', fontsize=14)
    ax.set_ylabel('F1 Score (%) on Full Test Set', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    
    if completed_iterations < planned_iterations:
        ax.text(0.02, 0.98, 'Legend:\n○ Complete data\n× Partial data (API quota hit)', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    annotation_interval = 2 if completed_iterations < planned_iterations else 3
    for i, (col_name, label, color) in enumerate(ratio_columns):
        if i % 2 == 0:
            for j, (x, y) in enumerate(zip(final_results_df['no_worst_examples'], 
                                         final_results_df[col_name])):
                if j % annotation_interval == 0 and y > 0:
                    ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                               xytext=(0,15), ha='center', fontsize=9, color=color)
    
    plt.tight_layout()

    plot_filename = os.path.join(results_dir, f"{results_filename}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {plot_filename}")
    plt.show()

    logger.info("\n" + "="*80)
    logger.info("MIXED RATIO ANALYSIS SUMMARY")
    logger.info("="*80)
    logger.info(f"Completed iterations: {completed_iterations}/{planned_iterations}")

    if completed_iterations < planned_iterations:
        logger.warning("="*60)
        logger.warning("EXPERIMENT INCOMPLETE")
        logger.warning("="*60)
        logger.warning("API quota was exceeded during execution")
        logger.warning(f"Results available for: {list(final_results_df['no_worst_examples'])}")
        logger.warning("To complete experiment:")
        logger.warning("  1. Wait for quota reset (typically 24 hours)")
        logger.warning("  2. Re-run script - will resume from disk cache")
        logger.warning("  3. Or use partial results for preliminary analysis")
        logger.warning("="*60)
    else:
        logger.info("All iterations completed successfully!")
    
    for col_name, label, _ in ratio_columns:
        if not final_results_df[col_name].empty:
            best_f1 = final_results_df[col_name].max()
            if best_f1 > 0:
                best_idx = final_results_df[col_name].idxmax()
                best_examples = final_results_df.loc[best_idx, 'no_worst_examples']
                logger.info(f"🏆 {label}: Best F1={best_f1:.1f}% with {best_examples} examples")
            else:
                logger.info(f"❌ {label}: No successful results")
    
    cache_files = list(label_generator.cache_dir.glob("*.json"))
    total_cached_labels = 0
    for cache_file in cache_files:
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            total_cached_labels += len(cache_data.get('labels', []))
        except:
            pass
    
    logger.info(f"💾 Total labels in persistent cache: {total_cached_labels}")
    logger.info(f"📁 Cache directory: {label_generator.cache_dir}")
    
    if completed_iterations < planned_iterations:
        logger.info(f"\n📋 Next Steps:")
        logger.info(f"   1. Wait for API quota reset (typically 24 hours)")
        logger.info(f"   2. Re-run this script - it will automatically resume from cache")
        logger.info(f"   3. Experiment will continue from where it left off")
        logger.info(f"   4. Or analyze partial results for preliminary insights")


if __name__ == "__main__":
    main()
