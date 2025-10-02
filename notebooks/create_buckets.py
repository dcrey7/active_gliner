#!/usr/bin/env python3
"""
Prepare MSE Confidence Buckets - CORRECTED VERSION
Generate MSE range-based buckets with equal MSE intervals
"""

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import warnings
import numpy as np
from pathlib import Path

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
from gliner import GLiNER


def calculate_mse(scores):
    """Calculate MSE from confidence scores"""
    if not scores:
        return 0.0
    squared_errors = [(1.0 - score)**2 for score in scores]
    return sum(squared_errors) / len(squared_errors)


def create_mse_range_buckets(sorted_examples, examples_per_bucket, logger):
    """
    Create 5 equal MSE range-based buckets
    
    Args:
        sorted_examples: All examples sorted by MSE (high to low)
        examples_per_bucket: Target number of examples per bucket
        logger: Logger instance
        
    Returns:
        List of 5 buckets with statistics
    """
    # Find actual MSE range in data
    max_mse = max(ex['mse'] for ex in sorted_examples)
    min_mse = min(ex['mse'] for ex in sorted_examples)
    
    # Avoid division by zero
    if min_mse == 0.0:
        min_mse = 0
    
    logger.info(f"MSE Range in data: {min_mse:.4f} to {max_mse:.4f}")
    
    # Calculate bucket width
    mse_range = max_mse - min_mse
    bucket_width = mse_range / 5
    
    logger.info(f"Bucket width: {bucket_width:.4f}")
    
    # Define 5 equal MSE range buckets
    bucket_ranges = []
    for i in range(5):
        lower = max_mse - (i + 1) * bucket_width
        upper = max_mse - i * bucket_width
        bucket_ranges.append((lower, upper))
    
    logger.info(f"\nMSE Ranges for 5 buckets:")
    for i, (lower, upper) in enumerate(bucket_ranges, 1):
        logger.info(f"   Bucket {i}: {lower:.4f} - {upper:.4f}")
    
    # Filter examples into buckets
    buckets = []
    
    for i, (lower, upper) in enumerate(bucket_ranges, 1):
        # Get all examples in this MSE range
        bucket_examples = [
            ex for ex in sorted_examples 
            if lower <= ex['mse'] <= upper
        ]
        
        # Take up to examples_per_bucket
        if len(bucket_examples) > examples_per_bucket:
            import random
            random.seed(42)  # For reproducibility
            bucket_examples = random.sample(bucket_examples, examples_per_bucket)
        
        # Calculate statistics
        if bucket_examples:
            mses = [ex['mse'] for ex in bucket_examples]
            confidences = [score for ex in bucket_examples for score in ex['scores']]
            
            bucket_info = {
                'bucket_number': i,
                'mse_range_definition': (lower, upper),
                'examples': bucket_examples,
                'num_examples': len(bucket_examples),
                'mse_range': (min(mses), max(mses)),
                'avg_mse': np.mean(mses),
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'avg_confidence_pct': np.mean(confidences) * 100 if confidences else 0.0,
                'total_entities': sum(len(ex['ner']) for ex in bucket_examples),
                'avg_entities': sum(len(ex['ner']) for ex in bucket_examples) / len(bucket_examples)
            }
        else:
            # Empty bucket
            bucket_info = {
                'bucket_number': i,
                'mse_range_definition': (lower, upper),
                'examples': [],
                'num_examples': 0,
                'mse_range': (0.0, 0.0),
                'avg_mse': 0.0,
                'avg_confidence': 0.0,
                'avg_confidence_pct': 0.0,
                'total_entities': 0,
                'avg_entities': 0.0
            }
        
        buckets.append(bucket_info)
    
    return buckets


def main():
    """Main function to prepare MSE buckets"""
    
    # ===============================================================================
    # Setup
    # ===============================================================================
    
    settings = Settings()
    settings.setup()
    logger = setup_logging(log_dir=str(settings.logs_dir))
    set_all_seeds(seed=settings.global_seed, logger=logger)
    device = setup_device(logger=logger)
    
    logger.info("="*80)
    logger.info("MSE BUCKET PREPARATION - RANGE-BASED")
    logger.info("="*80)
    
    # User configuration
    EXAMPLES_PER_BUCKET = 500  # Adjust this as needed
    
    # ===============================================================================
    # Load Training Data
    # ===============================================================================
    
    train_data_path = settings.data_path / settings.train_file
    labels_path = settings.data_path / settings.labels_file
    
    if not (train_data_path.exists() and labels_path.exists()):
        raise FileNotFoundError("Train data or labels file not found!")
    
    train_data, entity_types = load_mit_dataset(str(train_data_path), str(labels_path), "train")
    logger.info(f"📊 Loaded train data: {len(train_data)} examples, {len(entity_types)} entity types")
    logger.info(f"📋 Entity types: {entity_types}")
    
    # ===============================================================================
    # Initialize GLiNER Base Model
    # ===============================================================================
    
    logger.info("🤖 Initializing GLiNER base model...")
    model = GLiNER.from_pretrained("knowledgator/modern-gliner-bi-large-v1.0")
    model.config.max_len = 8192
    
    if hasattr(model.data_processor, 'transformer_tokenizer'):
        model.data_processor.transformer_tokenizer.model_max_length = 8192
    
    model.eval()
    model.to(device)
    logger.info("✅ GLiNER model loaded and ready")
    
    # ===============================================================================
    # Run Evaluation to Get Predictions and Scores
    # ===============================================================================
    
    logger.info("📊 Running evaluation on training set to get predictions and scores...")
    
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
    
    # Extract predictions with scores
    all_predictions = train_results["all_predictions"]
    logger.info(f"✅ Got predictions for {len(all_predictions)} examples")
    
    # ===============================================================================
    # Calculate MSE and Sort
    # ===============================================================================
    
    logger.info("📈 Calculating MSE for all examples...")
    
    # Calculate MSE for each example
    for example in all_predictions:
        mse = calculate_mse(example['scores'])
        example['mse'] = mse
    
    # Sort by MSE (highest to lowest)
    sorted_examples = sorted(all_predictions, key=lambda x: x['mse'], reverse=True)
    
    logger.info(f"✅ Sorted {len(sorted_examples)} examples by MSE")
    logger.info(f"   Highest MSE: {sorted_examples[0]['mse']:.4f}")
    logger.info(f"   Lowest MSE: {sorted_examples[-1]['mse']:.4f}")
    logger.info(f"   Average MSE: {np.mean([ex['mse'] for ex in sorted_examples]):.4f}")
    
    # ===============================================================================
    # Save All Sorted Examples
    # ===============================================================================
    
    results_dir = Path("../results/data")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_sorted_file = results_dir / "all_mse_sorted_examples.json"
    logger.info(f"💾 Saving all sorted examples to: {all_sorted_file}")
    
    with open(all_sorted_file, 'w') as f:
        json.dump(sorted_examples, f, indent=2)
    
    logger.info(f"✅ Saved {len(sorted_examples)} sorted examples")
    
    # ===============================================================================
    # Create MSE Range-Based Buckets
    # ===============================================================================
    
    logger.info(f"\n📦 Creating 5 MSE range-based buckets with up to {EXAMPLES_PER_BUCKET} examples each...")
    
    buckets = create_mse_range_buckets(sorted_examples, EXAMPLES_PER_BUCKET, logger)
    
    # ===============================================================================
    # Save Each Bucket and Log Statistics
    # ===============================================================================
    
    logger.info("\n" + "="*80)
    logger.info("BUCKET STATISTICS")
    logger.info("="*80)
    
    for bucket in buckets:
        bucket_num = bucket['bucket_number']
        
        # Save bucket examples
        bucket_file = results_dir / f"bucket_{bucket_num}_mse_{bucket['num_examples']}.json"
        
        with open(bucket_file, 'w') as f:
            json.dump(bucket['examples'], f, indent=2)
        
        logger.info(f"\nBucket {bucket_num}:")
        logger.info(f"   File: {bucket_file.name}")
        logger.info(f"   MSE range definition: {bucket['mse_range_definition'][0]:.4f} - {bucket['mse_range_definition'][1]:.4f}")
        logger.info(f"   Examples found: {bucket['num_examples']}")
        
        if bucket['num_examples'] > 0:
            logger.info(f"   Actual MSE range: {bucket['mse_range'][0]:.4f} - {bucket['mse_range'][1]:.4f}")
            logger.info(f"   Average MSE: {bucket['avg_mse']:.4f}")
            logger.info(f"   Average confidence: {bucket['avg_confidence']:.4f} ({bucket['avg_confidence_pct']:.1f}%)")
            logger.info(f"   Total entities: {bucket['total_entities']}")
            logger.info(f"   Avg entities per example: {bucket['avg_entities']:.1f}")
        else:
            logger.info(f"   ⚠️  No examples found in this MSE range")
    
    # ===============================================================================
    # Save Bucket Summary
    # ===============================================================================
    
    summary = {
        'total_examples': len(sorted_examples),
        'target_examples_per_bucket': EXAMPLES_PER_BUCKET,
        'num_buckets': 5,
        'bucketing_method': 'mse_range_based',
        'bucket_summaries': [
            {
                'bucket_number': b['bucket_number'],
                'mse_range_definition': b['mse_range_definition'],
                'num_examples': b['num_examples'],
                'mse_range': b['mse_range'],
                'avg_mse': b['avg_mse'],
                'avg_confidence_pct': b['avg_confidence_pct'],
                'avg_entities': b['avg_entities']
            }
            for b in buckets
        ]
    }
    
    summary_file = results_dir / "bucket_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n💾 Saved bucket summary to: {summary_file}")
    
    # ===============================================================================
    # Distribution Analysis
    # ===============================================================================
    
    logger.info("\n" + "="*80)
    logger.info("DISTRIBUTION ANALYSIS")
    logger.info("="*80)
    
    total_bucketed = sum(b['num_examples'] for b in buckets)
    logger.info(f"Total examples bucketed: {total_bucketed}/{EXAMPLES_PER_BUCKET * 5} requested")
    
    for bucket in buckets:
        if bucket['num_examples'] > 0:
            pct = (bucket['num_examples'] / EXAMPLES_PER_BUCKET) * 100
            logger.info(f"Bucket {bucket['bucket_number']}: {bucket['num_examples']}/150 ({pct:.1f}%)")
        else:
            logger.info(f"Bucket {bucket['bucket_number']}: Empty ⚠️")
    
    # ===============================================================================
    # Cleanup
    # ===============================================================================
    
    del model
    torch.cuda.empty_cache()
    
    logger.info("\n" + "="*80)
    logger.info("✅ MSE BUCKET PREPARATION COMPLETED")
    logger.info("="*80)
    logger.info(f"📁 All files saved to: {results_dir}")
    logger.info(f"📊 Total buckets created: 5")
    logger.info(f"📋 Ready for experiment!")


if __name__ == "__main__":
    main()