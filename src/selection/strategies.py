"""
Active learning selection strategies - only your original function
"""

from typing import List, Dict, Any
import logging


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
