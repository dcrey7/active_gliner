"""
Analysis pipeline management - handles domain analysis with batch caching
Keeps analysis independent from synthetic data generation
"""

from typing import List, Dict, Any, Optional
import logging
from .batch_analyzer import analyze_domain_with_batch_caching
from .summarizer import combine_all_batch_results, create_final_summary_with_retry


def get_or_create_analysis(num_examples: int, low_confidence_examples: List[Dict], 
                         entity_types: List[str], cache_manager, settings,
                         batch_size: int = 10, skip_analysis: bool = False,
                         logger: Optional[logging.Logger] = None) -> Optional[Dict]:
    """
    Get analysis from cache or create new analysis with batch caching optimization
    Exact copy of your original function but using dependency injection
    
    Args:
        num_examples: Number of corrected examples (cache key)
        low_confidence_examples: List of low confidence examples to analyze
        entity_types: List of entity types
        cache_manager: CacheManager instance
        settings: Settings object
        batch_size: Batch size for analysis
        skip_analysis: Whether to skip analysis entirely
        logger: Optional logger
        
    Returns:
        Dictionary with analysis results or None if failed
    """
    
    # Handle zero examples case - exact logic from your original
    if num_examples == 0:
        if logger:
            logger.info(f"⏭️ Skipping analysis pipeline (num_examples=0 - no corrected examples to analyze)")
        return {'final_summary': None, 'combined_result': None, 'all_batch_results': None}
    
    if skip_analysis:
        if logger:
            logger.info(f"⏭️ Skipping analysis pipeline (skip_analysis=True)")
        return {'final_summary': None, 'combined_result': None, 'all_batch_results': None}
    
    cache_key = num_examples
    
    # Check cache first
    cached_result = cache_manager.get_final_summary(cache_key)
    if cached_result:
        if logger:
            logger.info(f"📋 Using cached final summary for {num_examples} examples")
        return cached_result
    
    if logger:
        logger.info(f"🔄 Creating new analysis for {num_examples} examples")
    
    # Process examples in batches WITH BATCH CACHING - exact flow from your original
    all_batch_results = analyze_domain_with_batch_caching(
        low_confidence_examples, entity_types, cache_manager, settings, batch_size, logger
    )
    
    if not all_batch_results:
        if logger:
            logger.error("❌❌❌ CRITICAL: Batch analysis failed after retries - STOPPING PIPELINE")
        return None
    
    # Combine all batch results - exact function call from your original
    combined_result = combine_all_batch_results(all_batch_results, entity_types, logger)
    
    # Create final summary WITH RETRY LOGIC - exact function call from your original
    final_summary = create_final_summary_with_retry(combined_result, settings, logger=logger)
    
    if not final_summary:
        if logger:
            logger.error("❌❌❌ CRITICAL: Final summary failed after retries - STOPPING PIPELINE")
        return None
    
    if logger:
        logger.info(f"📋 Analysis Summary: {final_summary.get('domain_summary', 'N/A')[:100]}...")
    
    # Cache the results at the final summary level - exact structure from your original
    analysis_result = {
        'final_summary': final_summary,
        'combined_result': combined_result,
        'all_batch_results': all_batch_results
    }
    
    cache_manager.set_final_summary(cache_key, analysis_result)
    
    if logger:
        logger.info(f"💾 Cached final summary for {num_examples} examples")
        cache_manager.log_cache_status(logger)
    
    return analysis_result


def get_analysis_status(cache_manager, logger: Optional[logging.Logger] = None) -> Dict:
    """
    Get current analysis pipeline status and cache statistics
    
    Args:
        cache_manager: CacheManager instance
        logger: Optional logger
        
    Returns:
        Dictionary with analysis status
    """
    stats = cache_manager.get_cache_stats()
    
    status = {
        'analysis_cached_configs': list(cache_manager.final_summary_cache.keys()),
        'total_batch_analyses': stats['batch_analysis_cache'],
        'analysis_ready': True
    }
    
    if logger:
        logger.info("Analysis Pipeline Status:")
        logger.info(f"  Analysis configs cached: {len(status['analysis_cached_configs'])}")
        logger.info(f"  Total batch analyses: {status['total_batch_analyses']}")
    
    return status