#!/usr/bin/env python3
"""
Generate High MSE Examples Script
Loads MIT movie data, evaluates GLiNER on train set, and saves high MSE examples
"""

import sys
import os
import json
import torch
import warnings

# Suppress warnings
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
from selection.strategies import get_highest_mse_examples_sorted, compare_selection_strategies
from gliner import GLiNER


def main():
    """Generate and save high MSE examples from MIT movie train set"""
    
    # ===============================================================================
    # 1. Setup and Configuration
    # ===============================================================================
    
    settings = Settings()
    settings.setup()
    logger = setup_logging(log_dir=str(settings.logs_dir))
    set_all_seeds(seed=settings.global_seed, logger=logger)
    device = setup_device(logger=logger)
    
    logger.info("="*80)
    logger.info("GENERATING HIGH MSE EXAMPLES FROM TRAIN SET")
    logger.info("="*80)
    
    # ===============================================================================
    # 2. Load MIT Movie Dataset
    # ===============================================================================
    
    train_data_path = settings.data_path / settings.train_file
    labels_path = settings.data_path / settings.labels_file
    
    if not (train_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Train data or labels file not found!")
    
    train_data, entity_types = load_mit_dataset(str(train_data_path), str(labels_path), "train")
    logger.info(f"📊 Loaded train data: {len(train_data)} examples, {len(entity_types)} entity types")
    logger.info(f"📋 Entity types: {entity_types}")
    
    # ===============================================================================
    # 3. Initialize GLiNER Model
    # ===============================================================================
    
    logger.info("🤖 Initializing GLiNER model...")
    model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
    model.config.max_len = 8192
    
    if hasattr(model.data_processor, 'transformer_tokenizer'):
        model.data_processor.transformer_tokenizer.model_max_length = 8192
    
    model.eval()
    model.to(device)
    logger.info("✅ GLiNER model loaded and ready")
    
    # ===============================================================================
    # 4. Evaluate GLiNER on Train Set
    # ===============================================================================
    
    logger.info("📊 Running enhanced evaluation on train set...")
    
    with torch.no_grad():
        train_results = enhanced_evaluate(
            model=model,
            data=train_data,
            entity_types=entity_types,
            threshold=0.5,
            batch_size=8,
            has_ground_truth=True,
            logger=logger
        )
    
    # Extract overall metrics
    overall_metrics = train_results["overall_metrics"]
    train_f1 = overall_metrics["overall_f1_pct"]
    train_confidence = overall_metrics["overall_confidence_pct"]
    
    logger.info("📋 Train Set Evaluation Results:")
    logger.info(f"   F1 Score: {train_f1:.2f}%")
    logger.info(f"   Confidence: {train_confidence:.2f}%")
    logger.info(f"   Total Examples: {overall_metrics['total_examples']}")
    
    # ===============================================================================
    # 5. Extract High MSE Examples
    # ===============================================================================
    
    logger.info("🔍 Extracting high MSE examples...")
    
    # Get 1000 examples with highest MSE
    high_mse_examples = get_highest_mse_examples_sorted(
        training_pool_results=train_results,
        n=2500,
        logger=logger
    )
    
    logger.info(f"✅ Extracted {len(high_mse_examples)} high MSE examples")
    
    # ===============================================================================
    # 6. Compare Selection Strategies
    # ===============================================================================
    
    logger.info("📊 Comparing selection strategies...")
    
    comparison_results = compare_selection_strategies(
        training_pool_results=train_results,
        n=100,  # Compare top 100 from each strategy
        logger=logger
    )
    
    # ===============================================================================
    # 7. Save Results
    # ===============================================================================
    
    # Create results directory
    results_dir = os.path.join("..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save high MSE examples
    mse_examples_file = os.path.join(results_dir, "high_mse_2500_examples.json")
    
    logger.info(f"💾 Saving high MSE examples to: {mse_examples_file}")
    
    with open(mse_examples_file, 'w') as f:
        json.dump(high_mse_examples, f, indent=2)
    
    logger.info(f"✅ Saved {len(high_mse_examples)} high MSE examples")
    
    # Save comparison results
    comparison_file = os.path.join(results_dir, "selection_strategy_comparison.json")
    
    logger.info(f"💾 Saving strategy comparison to: {comparison_file}")
    
    comparison_summary = {
        'train_evaluation_metrics': {
            'f1_score': train_f1,
            'confidence': train_confidence,
            'total_examples': overall_metrics['total_examples']
        },
        'strategy_comparison': comparison_results,
        'high_mse_examples_count': len(high_mse_examples)
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    logger.info(f"✅ Saved strategy comparison results")
    
    # ===============================================================================
    # 8. Summary Statistics
    # ===============================================================================
    
    logger.info("="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    # Calculate statistics for high MSE examples
    all_scores = []
    all_mses = []
    example_lengths = []
    
    for example in high_mse_examples:
        scores = example.get('scores', [])
        tokenized_text = example.get('tokenized_text', [])
        
        if scores:
            all_scores.extend(scores)
            mse = sum((1.0 - score)**2 for score in scores) / len(scores)
            all_mses.append(mse)
        
        example_lengths.append(len(tokenized_text))
    
    if all_scores and all_mses:
        import numpy as np
        logger.info(f"High MSE Examples Statistics:")
        logger.info(f"   Average Confidence: {np.mean(all_scores):.3f}")
        logger.info(f"   Minimum Confidence: {min(all_scores):.3f}")
        logger.info(f"   Average MSE: {np.mean(all_mses):.3f}")
        logger.info(f"   Maximum MSE: {max(all_mses):.3f}")
        logger.info(f"   Average Text Length: {np.mean(example_lengths):.1f} tokens")
    
    # Show some examples
    logger.info("📋 Sample High MSE Examples:")
    for i, example in enumerate(high_mse_examples[:3]):
        scores = example.get('scores', [])
        text_preview = ' '.join(example['tokenized_text'][:10])
        if scores:
            mse = sum((1.0 - score)**2 for score in scores) / len(scores)
            min_score = min(scores)
            avg_score = np.mean(scores)
            logger.info(f"   Example {i+1}: MSE={mse:.3f}, min={min_score:.3f}, avg={avg_score:.3f}")
            logger.info(f"      Text: '{text_preview}...'")
    
    logger.info("="*80)
    logger.info("HIGH MSE EXAMPLES GENERATION COMPLETED")
    logger.info("="*80)
    logger.info(f"📁 Files saved:")
    logger.info(f"   - High MSE examples: {mse_examples_file}")
    logger.info(f"   - Strategy comparison: {comparison_file}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()