#!/usr/bin/env python3
"""
Confidence Analysis Script 1: Base Model Performance on Difficult Examples
Tests GLiNER base model vs LLM performance on worst confidence example subsets
WITH SIMPLE CACHING using list cache (similar to Script 2)

Similar to test9_gemma.py but with varying subset sizes and caching
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Add src path
src_path = os.path.join(os.path.dirname(os.getcwd()), 'src')
sys.path.append(src_path)

# Import modules
from config.settings import Settings
from utils.logging import setup_logging
from utils.reproducibility import set_all_seeds
from utils.device import setup_device
from data.loader import load_mit_dataset
from evaluation.evaluator import enhanced_evaluate
from evaluation.llm_evaluator import LLMEvaluationPipeline, LLMModelWrapper


def main():
    """Confidence Analysis: Base Model Performance on Difficult Examples"""
    
    # ===============================================================================
    # Setup
    # ===============================================================================
    
    settings = Settings()
    settings.setup()
    logger = setup_logging(log_dir=str(settings.logs_dir))
    set_all_seeds(seed=settings.global_seed, logger=logger)
    device = setup_device(logger=logger)
    
    # Model Configuration  
    MODEL_TYPE = "ollama"
    MODEL_NAME = "gemma3:12b"
    
    # Load dataset
    test_data_path = settings.data_path / settings.test_file
    labels_path = settings.data_path / settings.labels_file
    
    if not (test_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Test data or labels file not found!")
    
    test_data, entity_types = load_mit_dataset(str(test_data_path), str(labels_path), "test")
    
    logger.info("="*60)
    logger.info("CONFIDENCE ANALYSIS: BASE MODEL PERFORMANCE (WITH CACHING)")
    logger.info("="*60)
    logger.info(f"Model: {MODEL_TYPE.upper()} - {MODEL_NAME}")
    logger.info(f"Entity types: {entity_types}")
    logger.info(f"Full test dataset: {len(test_data)} examples")
    
    # Load pre-saved low confidence examples
    logger.info("📂 Loading pre-saved low confidence examples...")
    with open('../results/low_score_1000_examples.json', 'r') as file:
        low_n = json.load(file)
    logger.info(f"📊 Loaded {len(low_n)} low confidence examples")
    
    # ===============================================================================
    # Initialize Simple Caching (Like Script 2)
    # ===============================================================================
    
    # Initialize evaluation cache - this persists across all experiments
    evaluation_cache = []
    
    # Initialize LLM evaluation pipeline
    evaluation_pipeline = LLMEvaluationPipeline(
        model_type=MODEL_TYPE,
        model_name=MODEL_NAME
    )
    
    # ===============================================================================
    # Experiment Configuration
    # ===============================================================================
    
    subset_sizes = [10, 25,50,75,100,150,250,500,750,800,1000]
    
    # Results storage
    results = {
        'no_worst_examples': [],
        'gliner_base_f1': [],
        'llm_f1': [],
        'gliner_confidence': [],
        'llm_confidence': [],
        'total_examples': []
    }
    
    logger.info(f"🔬 Testing subset sizes: {subset_sizes}")
    logger.info(f"Total iterations: {len(subset_sizes)}")
    logger.info(f"📦 Evaluation cache initialized: {len(evaluation_cache)} examples")
    
    # ===============================================================================
    # Main Experiment Loop
    # ===============================================================================
    
    logger.info("🚀 Starting Base Performance Analysis...")
    logger.info("-" * 60)
    
    for n_examples in subset_sizes:
        logger.info(f"\n📊 Testing with {n_examples} worst confidence examples")
        
        # Get subset of worst confidence examples
        subset_examples = low_n[:n_examples]
        logger.info(f"Subset size: {len(subset_examples)} examples")
        
        # ===============================================================================
        # GLiNER Base Model Evaluation
        # ===============================================================================
        
        logger.info("🔵 Evaluating GLiNER Base Model...")
        
        # Load base GLiNER model
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
        import torch
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        # ===============================================================================
        # LLM Evaluation WITH SIMPLE CACHING
        # ===============================================================================
        
        logger.info("🔴 Evaluating LLM Model (with caching)...")
        
        # Generate LLM predictions on subset WITH CACHING
        # Use the same caching mechanism as Script 2 - simple list cache
        llm_predictions = evaluation_pipeline.evaluate_dataset(
            test_data=subset_examples,
            entity_types=entity_types,
            evaluation_cache=evaluation_cache  # Pass the persistent cache
        )
        
        logger.info(f"💾 Evaluation cache now contains: {len(evaluation_cache)} total examples")
        
        # Evaluate LLM predictions
        model_wrapper = LLMModelWrapper(llm_predictions)
        llm_results = enhanced_evaluate(
            model=model_wrapper,
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
        
        logger.info(f"✅ Results stored for {n_examples} examples")
        logger.info(f"📦 Cache efficiency: {len(evaluation_cache)} total evaluations available for reuse")
        
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
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
    results_path = f"../results/gemma/confidence_base_performance.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"💾 Results saved to: {results_path}")
    
    # ===============================================================================
    # Visualization
    # ===============================================================================
    
    logger.info("📈 Generating Trend Line Plot...")
    
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
        label='LLM (Gemma3:12b)', color='red', alpha=0.8
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
    plot_path = f"../results/gemma/confidence_base_performance_trend.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Plot saved to: {plot_path}")
    plt.show()
    
    # ===============================================================================
    # Summary with Cache Statistics
    # ===============================================================================
    
    logger.info("\n🎉 Base Performance Analysis completed successfully!")
    logger.info(f"📋 Best GLiNER F1: {max(results['gliner_base_f1']):.2f}% on {results['no_worst_examples'][results['gliner_base_f1'].index(max(results['gliner_base_f1']))]} examples")
    logger.info(f"📋 Best LLM F1: {max(results['llm_f1']):.2f}% on {results['no_worst_examples'][results['llm_f1'].index(max(results['llm_f1']))]} examples")
    logger.info(f"💾 Total LLM evaluations cached for reuse: {len(evaluation_cache)} examples")
    
    # Calculate cache efficiency
    max_subset_size = max(subset_sizes)
    cache_efficiency = (len(evaluation_cache) / (max_subset_size * len(subset_sizes))) * 100 if max_subset_size > 0 else 0
    logger.info(f"📊 Cache efficiency: {cache_efficiency:.1f}% (saved {len(subset_sizes) * max_subset_size - len(evaluation_cache)} redundant evaluations)")


if __name__ == "__main__":
    main()