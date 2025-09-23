"""
Active learning selection strategies - original + MSE-based selection
"""

from typing import List, Dict, Any
import logging
import numpy as np


def get_lowest_score_examples_sorted(training_pool_results: Dict, n: int = 5,
                                   logger: logging.Logger = None) -> List[Dict]:
    """
    Get n examples with lowest minimum scores exactly like your original function
    
    Args:
        training_pool_results: Results from enhanced_evaluate on training pool
        n: Number of examples to return
        logger: Optional logger
        
    Returns:
        List of examples sorted by lowest confidence scores
    """
    if n == 0:
        if logger:
            logger.info("Requested 0 examples - returning empty list")
        return []
        
    if logger:
        logger.info(f"Extracting {n} lowest confidence examples from training pool evaluation results...")
    
    examples = training_pool_results['all_predictions']
    
    # Sort examples by their minimum score
    sorted_examples = sorted(
        examples, 
        key=lambda x: min(x['scores']) if x['scores'] else 1.0
    )
    
    if logger:
        logger.info(f"Total examples available: {len(examples)}")
        logger.info(f"Returning {min(n, len(sorted_examples))} lowest confidence examples")
        
        # Log sample of lowest confidence examples
        for i, ex in enumerate(sorted_examples[:min(3, n)]):
            min_score = min(ex['scores']) if ex['scores'] else 0.0
            text_preview = ' '.join(ex['tokenized_text'][:8])
            logger.info(f"  Example {i+1}: score={min_score:.3f}, text='{text_preview}...'")
    
    return sorted_examples[:n]


def get_highest_mse_examples_sorted(training_pool_results: Dict, n: int = 5,
                                   logger: logging.Logger = None) -> List[Dict]:
    """
    Get n examples with highest Mean Squared Error of confidence scores
    MSE = Σ(1 - confidence_i)² / num_entities
    
    This captures overall model uncertainty rather than just worst single prediction
    
    Args:
        training_pool_results: Results from enhanced_evaluate on training pool
        n: Number of examples to return
        logger: Optional logger
        
    Returns:
        List of examples sorted by highest MSE (most systematically uncertain)
    """
    if n == 0:
        if logger:
            logger.info("Requested 0 examples - returning empty list")
        return []
        
    if logger:
        logger.info(f"Extracting {n} highest MSE confidence examples from training pool evaluation results...")
    
    examples = training_pool_results['all_predictions']
    
    # Calculate MSE for each example and sort by highest MSE (following same pattern as original)
    def calculate_mse(x):
        scores = x.get('scores', [])
        if not scores:
            # No predictions = treat as low priority (like original function treats as high confidence)
            return 0.0
        else:
            # Calculate MSE: mean of squared errors from perfect confidence (1.0)
            squared_errors = [(1.0 - score)**2 for score in scores]
            return sum(squared_errors) / len(squared_errors)
    
    # Sort by highest MSE (reverse=True for highest first, just like min uses ascending)
    sorted_examples = sorted(
        examples, 
        key=calculate_mse,
        reverse=True
    )
    
    if logger:
        logger.info(f"Total examples available: {len(examples)}")
        logger.info(f"Returning {min(n, len(sorted_examples))} highest MSE examples")
        
        # Log sample of highest MSE examples
        for i, ex in enumerate(sorted_examples[:min(3, n)]):
            scores = ex.get('scores', [])
            if scores:
                mse = calculate_mse(ex)
                min_score = min(scores)
                avg_score = sum(scores) / len(scores)
                text_preview = ' '.join(ex['tokenized_text'][:8])
                logger.info(f"  Example {i+1}: MSE={mse:.3f}, min={min_score:.3f}, avg={avg_score:.3f}, text='{text_preview}...'")
            else:
                text_preview = ' '.join(ex['tokenized_text'][:8])
                logger.info(f"  Example {i+1}: No predictions, text='{text_preview}...'")
    
    return sorted_examples[:n]


def compare_selection_strategies(training_pool_results: Dict, n: int = 10,
                               logger: logging.Logger = None) -> Dict:
    """
    Compare minimum score vs MSE selection strategies
    
    Args:
        training_pool_results: Results from enhanced_evaluate on training pool
        n: Number of examples to compare
        logger: Optional logger
        
    Returns:
        Dictionary with comparison results
    """
    if logger:
        logger.info(f"Comparing selection strategies for top {n} examples")
    
    # Get examples from both strategies
    min_examples = get_lowest_score_examples_sorted(training_pool_results, n, logger=None)
    mse_examples = get_highest_mse_examples_sorted(training_pool_results, n, logger=None)
    
    # Calculate overlap
    min_texts = set(' '.join(ex['tokenized_text']) for ex in min_examples)
    mse_texts = set(' '.join(ex['tokenized_text']) for ex in mse_examples)
    overlap = len(min_texts & mse_texts)
    
    # Calculate statistics for each strategy
    def get_stats(examples):
        all_scores = []
        all_mses = []
        for ex in examples:
            scores = ex.get('scores', [])
            if scores:
                all_scores.extend(scores)
                mse = sum((1.0 - score)**2 for score in scores) / len(scores)
                all_mses.append(mse)
        
        return {
            'avg_confidence': np.mean(all_scores) if all_scores else 0.0,
            'min_confidence': min(all_scores) if all_scores else 0.0,
            'avg_mse': np.mean(all_mses) if all_mses else 0.0,
            'num_examples': len(examples)
        }
    
    min_stats = get_stats(min_examples)
    mse_stats = get_stats(mse_examples)
    
    comparison = {
        'overlap_count': overlap,
        'overlap_percentage': (overlap / n) * 100 if n > 0 else 0,
        'minimum_strategy_stats': min_stats,
        'mse_strategy_stats': mse_stats
    }
    
    if logger:
        logger.info("Strategy Comparison Results:")
        logger.info(f"  Overlap: {overlap}/{n} examples ({comparison['overlap_percentage']:.1f}%)")
        logger.info(f"  Minimum Strategy - Avg Conf: {min_stats['avg_confidence']:.3f}, Avg MSE: {min_stats['avg_mse']:.3f}")
        logger.info(f"  MSE Strategy - Avg Conf: {mse_stats['avg_confidence']:.3f}, Avg MSE: {mse_stats['avg_mse']:.3f}")
    
    return comparison