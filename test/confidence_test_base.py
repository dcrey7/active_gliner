#!/usr/bin/env python3
"""
Confidence Analysis: Base Model Performance on Difficult Examples
Compares GLiNER base model vs LLM on worst confidence subsets

Clean architecture using:
- GLONER.default() for base GLiNER model  
- create_predictor() for LLM predictions (evaluation mode)
- enhanced_evaluate() for both GLiNER and LLM
- MockGLiNERModel to wrap LLM predictions for evaluation
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

# Config and utilities
from config.settings import Settings
from config.constants import GLOBAL_SEED
from utils.logging import setup_logging
from utils.reproducibility import set_all_seeds
from utils.device import setup_device
from utils.memory import cleanup_memory

# Data loading
from data.loader import load_mit_dataset

# Model and evaluation
from models.gloner import GLONER
from evaluation.evaluator import enhanced_evaluate
from generation.inference_helper import create_llm_gliner_wrapper

# LLM inference (unified class)
from generation.llm_inference import create_predictor


def main():
    """Confidence Analysis: Base Model Performance"""

    # ===============================================================================
    # Setup and Configuration
    # ===============================================================================

    settings = Settings()
    settings.setup()
    logger = setup_logging(log_dir=str(settings.logs_dir))
    set_all_seeds(seed=GLOBAL_SEED, logger=logger)
    device = setup_device(logger=logger)

    # LLM Configuration
    LLM_BACKEND = "ollama"
    LLM_MODEL = "gemma3:12b"
    USE_STRUCTURED = False

    # Load dataset
    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file

    if not (test_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Test data or labels file not found!")

    test_data, entity_types = load_mit_dataset(
        str(test_data_path), 
        str(labels_path), 
        "test"
    )

    logger.info("="*80)
    logger.info("CONFIDENCE ANALYSIS: BASE MODEL PERFORMANCE ON DIFFICULT EXAMPLES")
    logger.info("="*80)
    logger.info(f"LLM Backend: {LLM_BACKEND.upper()}")
    logger.info(f"LLM Model: {LLM_MODEL}")
    logger.info(f"Structured Output: {USE_STRUCTURED}")
    logger.info(f"Entity types ({len(entity_types)}): {entity_types}")
    logger.info(f"Full test dataset: {len(test_data)} examples")

    # Load pre-saved low confidence examples
    logger.info("\nLoading pre-saved low confidence examples...")
    low_conf_file = os.path.join(
        os.path.dirname(__file__), 
        '../results/high_mse_2500_examples.json'
    )
    with open(low_conf_file, 'r') as f:
        low_confidence_examples = json.load(f)
    
    logger.info(f"Loaded {len(low_confidence_examples)} low confidence examples")

    # ===============================================================================
    # Initialize Models
    # ===============================================================================

    # GLiNER Base Model (no LoRA adapter, just base model)
    logger.info("\nInitializing GLiNER base model...")
    gliner_base = GLONER.default(logger=logger)
    gliner_base.to(device)
    logger.info("GLiNER base model loaded successfully")

    # LLM Predictor (evaluation mode - preserves all indices)
    logger.info(f"\nInitializing LLM predictor ({LLM_BACKEND})...")
    llm_predictor = create_predictor(
        backend_type=LLM_BACKEND,
        model_name=LLM_MODEL,
        entity_types=entity_types,
        cache_type='disk',  # Persistent cache for evaluation
        use_structured_output=USE_STRUCTURED,
        logger=logger
    )
    logger.info("LLM predictor initialized successfully")

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

    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info(f"{'='*80}")
    logger.info(f"Subset sizes to test: {subset_sizes}")
    logger.info(f"Total iterations: {len(subset_sizes)}")
    logger.info(f"Evaluation: GLiNER base vs LLM on each subset")
    logger.info(f"{'='*80}\n")

    # ===============================================================================
    # Main Experiment Loop
    # ===============================================================================

    logger.info("Starting Base Performance Analysis...")
    logger.info("-" * 80)

    for n_examples in subset_sizes:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING WITH {n_examples} WORST CONFIDENCE EXAMPLES")
        logger.info(f"{'='*80}\n")

        # Get subset of worst confidence examples
        subset_examples = low_confidence_examples[:n_examples]
        logger.info(f"Subset size: {len(subset_examples)} examples")

        # ===============================================================================
        # GLiNER Base Model Evaluation
        # ===============================================================================

        logger.info("\n" + "-"*80)
        logger.info("EVALUATING GLINER BASE MODEL")
        logger.info("-"*80)

        gliner_results = enhanced_evaluate(
            model=gliner_base,
            data=subset_examples,
            entity_types=entity_types,
            threshold=0.5,
            batch_size=8,
            has_ground_truth=True,
            logger=logger
        )

        gliner_f1 = gliner_results["overall_metrics"]["overall_f1_pct"]
        gliner_conf = gliner_results["overall_metrics"]["overall_confidence_pct"]

        logger.info(f"\n✅ GLiNER Base Results:")
        logger.info(f"   F1 Score: {gliner_f1:.2f}%")
        logger.info(f"   Confidence: {gliner_conf:.2f}%")

        # Cleanup GLiNER model memory
        cleanup_memory()

        # ===============================================================================
        # LLM Evaluation
        # ===============================================================================

        logger.info("\n" + "-"*80)
        logger.info("EVALUATING LLM PREDICTIONS")
        logger.info("-"*80)

        # Step 1: Generate LLM predictions (evaluation mode - preserves indices)
        logger.info("\nGenerating LLM predictions...")
        llm_results_dict = llm_predictor.generate(
            examples=subset_examples,
            entity_types=entity_types,
            num_samples=n_examples,
            verbose=True
        )

        llm_predictions = llm_results_dict['all_labels']
        logger.info(f"Generated {len(llm_predictions)} predictions")

        # Step 2: Wrap LLM predictions in mock model for enhanced_evaluate
        logger.info("\nWrapping LLM predictions for evaluation...")
        mock_llm_model = create_llm_gliner_wrapper(llm_predictions)

        # Step 3: Evaluate using enhanced_evaluate (same as GLiNER)
        logger.info("\nEvaluating LLM predictions against ground truth...")
        llm_eval_results = enhanced_evaluate(
            model=mock_llm_model,
            data=subset_examples,
            entity_types=entity_types,
            threshold=0.5,  # Not used for LLM (no confidence scores)
            batch_size=8,
            has_ground_truth=True,
            logger=logger
        )

        llm_f1 = llm_eval_results["overall_metrics"]["overall_f1_pct"]
        llm_conf = llm_eval_results["overall_metrics"]["overall_confidence_pct"]

        logger.info(f"\n✅ LLM Results:")
        logger.info(f"   F1 Score: {llm_f1:.2f}%")
        logger.info(f"   Confidence: {llm_conf:.2f}%")

        # ===============================================================================
        # Store Results
        # ===============================================================================

        results['no_worst_examples'].append(n_examples)
        results['gliner_base_f1'].append(gliner_f1)
        results['llm_f1'].append(llm_f1)
        results['gliner_confidence'].append(gliner_conf)
        results['llm_confidence'].append(llm_conf)
        results['total_examples'].append(n_examples)

        logger.info(f"\n{'='*80}")
        logger.info(f"COMPARISON FOR {n_examples} EXAMPLES")
        logger.info(f"{'='*80}")
        logger.info(f"GLiNER Base F1: {gliner_f1:.2f}%")
        logger.info(f"LLM F1:         {llm_f1:.2f}%")
        logger.info(f"Winner:         {'LLM' if llm_f1 > gliner_f1 else 'GLiNER' if gliner_f1 > llm_f1 else 'TIE'}")
        logger.info(f"Difference:     {abs(llm_f1 - gliner_f1):.2f}%")
        logger.info(f"{'='*80}\n")

        # Cleanup
        cleanup_memory()

    # ===============================================================================
    # Results Analysis
    # ===============================================================================

    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*80)

    # Create DataFrame
    final_results_df = pd.DataFrame(results)

    # Display results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    logger.info("\n" + final_results_df.to_string(index=False))
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), f'../results/{LLM_BACKEND}')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(
        results_dir, 
        f"confidence_base_performance_{LLM_MODEL.replace(':', '_')}.csv"
    )
    final_results_df.to_csv(results_file, index=False)
    logger.info(f"\n✅ Results saved to: {results_file}")

    # ===============================================================================
    # Visualization
    # ===============================================================================

    logger.info("\nGenerating performance comparison plot...")

    plt.style.use('default')
    sns.set_palette("husl")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot GLiNER Base
    ax.plot(
        final_results_df['no_worst_examples'], 
        final_results_df['gliner_base_f1'],
        marker='o', markersize=10, linewidth=3,
        label='GLiNER Base Model', color='blue', alpha=0.8
    )

    # Plot LLM
    ax.plot(
        final_results_df['no_worst_examples'], 
        final_results_df['llm_f1'],
        marker='s', markersize=10, linewidth=3,
        label=f'LLM ({LLM_MODEL})', color='green', alpha=0.8
    )

    # Formatting
    ax.set_title(
        'Base Model Performance: GLiNER vs LLM on Difficult Examples',
        fontsize=16, fontweight='bold', pad=20
    )
    ax.set_xlabel('Number of Worst Confidence Examples', fontsize=14)
    ax.set_ylabel('F1 Score (%)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')

    # Add value annotations
    for i, (x, y1, y2) in enumerate(zip(
        final_results_df['no_worst_examples'],
        final_results_df['gliner_base_f1'],
        final_results_df['llm_f1']
    )):
        ax.annotate(f'{y1:.1f}%', (x, y1), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=10, color='blue')
        ax.annotate(f'{y2:.1f}%', (x, y2), textcoords="offset points",
                   xytext=(0,-15), ha='center', fontsize=10, color='green')

    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(
        results_dir, 
        f"confidence_base_performance_trend_{LLM_MODEL.replace(':', '_')}.png"
    )
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Plot saved to: {plot_file}")
    plt.close()

    # ===============================================================================
    # Summary Statistics
    # ===============================================================================

    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    logger.info(f"Average GLiNER F1: {final_results_df['gliner_base_f1'].mean():.2f}%")
    logger.info(f"Average LLM F1:    {final_results_df['llm_f1'].mean():.2f}%")
    logger.info(f"Best GLiNER F1:    {final_results_df['gliner_base_f1'].max():.2f}%")
    logger.info(f"Best LLM F1:       {final_results_df['llm_f1'].max():.2f}%")
    
    wins_gliner = (final_results_df['gliner_base_f1'] > final_results_df['llm_f1']).sum()
    wins_llm = (final_results_df['llm_f1'] > final_results_df['gliner_base_f1']).sum()
    ties = (final_results_df['gliner_base_f1'] == final_results_df['llm_f1']).sum()
    
    logger.info(f"\nHead-to-Head:")
    logger.info(f"  GLiNER wins: {wins_gliner}")
    logger.info(f"  LLM wins:    {wins_llm}")
    logger.info(f"  Ties:        {ties}")
    logger.info("="*80)

    logger.info("\n✅ Base Performance Analysis completed successfully!")


if __name__ == "__main__":
    main()
