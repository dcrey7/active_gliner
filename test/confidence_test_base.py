#!/usr/bin/env python3
"""
Confidence Analysis Script 1: Base Model Performance on Difficult Examples
Tests GLiNER base model vs LLM performance on worst confidence example subsets

Uses new abstractions:
- GLONER for model loading
- create_ner_evaluator for LLM evaluation
- enhanced_evaluate for GLiNER evaluation
- DiskCache for persistent caching
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src path
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.append(src_path)

# Import with new abstractions
from config import Settings, GLOBAL_SEED
from utils import setup_logging, set_all_seeds, setup_device, cleanup_memory
from data import load_mit_dataset
from evaluation import enhanced_evaluate, create_ner_evaluator
from models.gloner import GLONER


def main():
    """Confidence Analysis: Base Model Performance on Difficult Examples"""

    # ===============================================================================
    # Setup
    # ===============================================================================

    settings = Settings()
    settings.setup()
    logger = setup_logging(log_dir=str(settings.logs_dir))
    set_all_seeds(seed=GLOBAL_SEED, logger=logger)
    device = setup_device(logger=logger)

    # LLM Configuration
    LLM_BACKEND = "cerebras"  # ollama, mistral, cerebras
    LLM_MODEL = "llama3.1-8b"
    USE_STRUCTURED = True  # Use structured output for Cerebras

    # Load dataset
    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file

    if not (test_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Test data or labels file not found!")

    test_data, entity_types = load_mit_dataset(str(test_data_path), str(labels_path), "test")

    logger.info("="*60)
    logger.info("CONFIDENCE ANALYSIS: BASE MODEL PERFORMANCE")
    logger.info("="*60)
    logger.info(f"LLM: {LLM_BACKEND.upper()} - {LLM_MODEL}")
    logger.info(f"Structured Output: {USE_STRUCTURED}")
    logger.info(f"Entity types: {entity_types}")
    logger.info(f"Full test dataset: {len(test_data)} examples")

    # Load pre-saved low confidence examples
    logger.info("Loading pre-saved low confidence examples...")
    low_conf_file = os.path.join(os.path.dirname(__file__), '../results/high_mse_2500_examples.json')
    with open(low_conf_file, 'r') as f:
        low_n = json.load(f)
    logger.info(f"Loaded {len(low_n)} low confidence examples")

    # ===============================================================================
    # Initialize LLM Evaluator with Caching
    # ===============================================================================

    logger.info("Initializing LLM evaluator with disk cache...")
    llm_evaluator = create_ner_evaluator(
        backend_type=LLM_BACKEND,
        entity_types=entity_types,
        model_name=LLM_MODEL,
        cache_type='disk',  # Use disk cache for persistence
        use_structured_output=USE_STRUCTURED
    )

    # ===============================================================================
    # Experiment Configuration
    # ===============================================================================

    subset_sizes = [2, 4, 8, 10]  # Small sizes for testing

    # Results storage
    results = {
        'no_worst_examples': [],
        'gliner_base_f1': [],
        'llm_f1': [],
        'gliner_confidence': [],
        'llm_confidence': [],
        'total_examples': []
    }

    logger.info(f"Testing subset sizes: {subset_sizes}")
    logger.info(f"Total iterations: {len(subset_sizes)}")

    # ===============================================================================
    # Main Experiment Loop
    # ===============================================================================

    logger.info("Starting Base Performance Analysis...")
    logger.info("-" * 60)

    for n_examples in subset_sizes:
        logger.info(f"\nTesting with {n_examples} worst confidence examples")

        # Get subset of worst confidence examples
        subset_examples = low_n[:n_examples]
        logger.info(f"Subset size: {len(subset_examples)} examples")

        # ===============================================================================
        # GLiNER Base Model Evaluation
        # ===============================================================================

        logger.info("Evaluating GLiNER Base Model...")

        # Load base GLiNER model (no LoRA, just base)
        base_model = GLONER.load_with_adapter(
            adapter_path=None,  # This will just load base model
            logger=None  # Suppress LoRA loading logs
        )
        # Actually, for base model, use direct GLiNER loading
        from gliner import GLiNER
        base_model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
        base_model.to(device)

        # Evaluate base GLiNER on subset
        gliner_results = enhanced_evaluate(
            model=base_model,
            data=subset_examples,
            entity_types=entity_types,
            threshold=0.5,
            batch_size=8,
            has_ground_truth=True,
            logger=logger
        )

        gliner_f1 = gliner_results["overall_metrics"]["overall_f1_pct"]
        gliner_conf = gliner_results["overall_metrics"]["overall_confidence_pct"]

        logger.info(f"GLiNER Base F1: {gliner_f1:.2f}%, Confidence: {gliner_conf:.2f}%")

        # Cleanup GLiNER model
        del base_model
        cleanup_memory()

        # ===============================================================================
        # LLM Evaluation with Caching
        # ===============================================================================

        logger.info("Evaluating LLM Model (with disk caching)...")

        # Evaluate using LLM evaluator (automatically caches to disk)
        llm_eval_results = llm_evaluator.evaluate(
            test_data=subset_examples,
            batch_size=1
        )

        # Get predictions and create wrapper for enhanced_evaluate
        llm_predictions = llm_eval_results['predictions']

        logger.info(f"Cache status: {len(llm_evaluator.cache.get_all())} examples cached")

        # Convert LLM predictions to format expected by enhanced_evaluate
        class LLMWrapper:
            """Wrapper to make LLM predictions work with enhanced_evaluate"""
            def __init__(self, predictions):
                self.predictions = predictions

            def run(self, texts, labels, flat_ner=True, threshold=0.5, batch_size=8):
                """Return pre-computed predictions"""
                return [[
                    {
                        'start': ent['start'],
                        'end': ent['end'],
                        'label': ent['label'],
                        'text': ent['text'],
                        'score': ent.get('score', 1.0)  # LLM doesn't provide scores
                    }
                    for ent in pred['ner']
                ] for pred in self.predictions]

        # Evaluate LLM predictions
        llm_wrapper = LLMWrapper(llm_predictions)
        llm_results = enhanced_evaluate(
            model=llm_wrapper,
            data=subset_examples,
            entity_types=entity_types,
            threshold=0.5,
            batch_size=8,
            has_ground_truth=True,
            logger=logger
        )

        llm_f1 = llm_results["overall_metrics"]["overall_f1_pct"]
        llm_conf = llm_results["overall_metrics"]["overall_confidence_pct"]

        logger.info(f"LLM F1: {llm_f1:.2f}%, Confidence: {llm_conf:.2f}%")

        # ===============================================================================
        # Store Results
        # ===============================================================================

        results['no_worst_examples'].append(n_examples)
        results['gliner_base_f1'].append(gliner_f1)
        results['llm_f1'].append(llm_f1)
        results['gliner_confidence'].append(gliner_conf)
        results['llm_confidence'].append(llm_conf)
        results['total_examples'].append(len(subset_examples))

        logger.info(f"Results stored for {n_examples} examples")

        # Cleanup
        cleanup_memory()

    # ===============================================================================
    # Results Analysis and Visualization
    # ===============================================================================

    logger.info("\n" + "="*60)
    logger.info("BASE PERFORMANCE ANALYSIS RESULTS")
    logger.info("="*60)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Configure pandas for full display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    logger.info("\n" + results_df.to_string(index=False))

    # Reset pandas display options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '../results', LLM_BACKEND)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"confidence_base_performance_{LLM_MODEL}.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to: {results_path}")

    # ===============================================================================
    # Visualization
    # ===============================================================================

    logger.info("Generating Trend Line Plot...")

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create trend line plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot GLiNER base performance
    ax.plot(
        results_df['no_worst_examples'], results_df['gliner_base_f1'],
        marker='o', markersize=8, linewidth=3,
        label='GLiNER Base Model', color='blue', alpha=0.8
    )

    # Plot LLM performance
    ax.plot(
        results_df['no_worst_examples'], results_df['llm_f1'],
        marker='s', markersize=8, linewidth=3,
        label=f'LLM ({LLM_BACKEND.title()}: {LLM_MODEL})', color='red', alpha=0.8
    )

    # Formatting
    ax.set_title('Base Model Performance on Worst Confidence Examples',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Worst Confidence Examples', fontsize=14)
    ax.set_ylabel('F1 Score (%)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Add value annotations
    for i, (x, y1, y2) in enumerate(zip(results_df['no_worst_examples'],
                                       results_df['gliner_base_f1'],
                                       results_df['llm_f1'])):
        ax.annotate(f'{y1:.1f}%', (x, y1), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=10, color='blue')
        ax.annotate(f'{y2:.1f}%', (x, y2), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=10, color='red')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(results_dir, f"confidence_base_performance_trend_{LLM_MODEL}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {plot_path}")
    plt.show()

    # ===============================================================================
    # Summary
    # ===============================================================================

    logger.info("\nBase Performance Analysis completed successfully!")
    logger.info(f"Best GLiNER F1: {max(results['gliner_base_f1']):.2f}% on {results['no_worst_examples'][results['gliner_base_f1'].index(max(results['gliner_base_f1']))]} examples")
    logger.info(f"Best LLM F1: {max(results['llm_f1']):.2f}% on {results['no_worst_examples'][results['llm_f1'].index(max(results['llm_f1']))]} examples")
    logger.info(f"Total LLM evaluations cached: {len(llm_evaluator.cache.get_all())} examples")


if __name__ == "__main__":
    main()
