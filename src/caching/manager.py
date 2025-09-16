"""
Cache management - replaces the global cache dictionaries from your original code
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging


class CacheManager:
    """
    Simple cache manager to replace global dictionaries
    Handles the three caches from your original code
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the three cache dictionaries like your original code
        self.batch_analysis_cache = {}      # Replaces BATCH_ANALYSIS_CACHE
        self.final_summary_cache = {}       # Replaces FINAL_SUMMARY_CACHE  
        self.synthetic_cache = {}           # Replaces SYNTHETIC_CACHE
        
        # Try to load existing caches
        self._load_caches()
    
    def get_batch_cache_key(self, batch_examples: List[Dict]) -> str:
        """
        Create cache key exactly like your original get_batch_cache_key function
        
        Args:
            batch_examples: List of example dictionaries
            
        Returns:
            MD5 hash string
        """
        # Create a string representation of the batch content
        batch_content = []
        for example in batch_examples:
            text = " ".join(example['tokenized_text'])
            ner = str(example['ner'])
            predictions = str(example['predictions'])
            scores = str(example['scores'])
            batch_content.append(f"{text}|{ner}|{predictions}|{scores}")
        
        # Create hash of the batch content
        batch_string = "||".join(batch_content)
        batch_hash = hashlib.md5(batch_string.encode()).hexdigest()
        return batch_hash
    
    def get_batch_analysis(self, batch_key: str) -> Optional[Dict]:
        """Get batch analysis from cache"""
        return self.batch_analysis_cache.get(batch_key)
    
    def set_batch_analysis(self, batch_key: str, analysis: Dict):
        """Store batch analysis in cache"""
        self.batch_analysis_cache[batch_key] = analysis
        self._save_cache('batch_analysis')
    
    def get_final_summary(self, num_examples: int) -> Optional[Dict]:
        """Get final summary from cache"""
        return self.final_summary_cache.get(num_examples)
    
    def set_final_summary(self, num_examples: int, summary: Dict):
        """Store final summary in cache"""
        self.final_summary_cache[num_examples] = summary
        self._save_cache('final_summary')
    
    def get_synthetic_data(self, num_examples: int) -> List[Dict]:
        """Get synthetic data from cache"""
        return self.synthetic_cache.get(num_examples, [])
    
    def add_synthetic_data(self, num_examples: int, new_data: List[Dict]):
        """Add synthetic data to cache"""
        if num_examples not in self.synthetic_cache:
            self.synthetic_cache[num_examples] = []
        self.synthetic_cache[num_examples].extend(new_data)
        self._save_cache('synthetic')
    
    def set_synthetic_data(self, num_examples: int, data: List[Dict]):
        """Set synthetic data in cache"""
        self.synthetic_cache[num_examples] = data
        self._save_cache('synthetic')
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics like your original log_cache_status function
        
        Returns:
            Dictionary with cache counts
        """
        stats = {
            'batch_analysis_cache': len(self.batch_analysis_cache),
            'final_summary_cache': len(self.final_summary_cache),
            'synthetic_cache': len(self.synthetic_cache)
        }
        
        # Add details about synthetic cache
        total_synthetic = sum(len(data) for data in self.synthetic_cache.values())
        stats['total_synthetic_examples'] = total_synthetic
        
        return stats
    
    def log_cache_status(self, logger: logging.Logger):
        """
        Log cache status exactly like your original function
        
        Args:
            logger: Logger instance
        """
        stats = self.get_cache_stats()
        
        logger.info("Cache Status:")
        for cache_type, count in stats.items():
            if cache_type != 'total_synthetic_examples':
                logger.info(f"  {cache_type}: {count}")
        
        if stats['total_synthetic_examples'] > 0:
            logger.info(f"  Total synthetic examples cached: {stats['total_synthetic_examples']}")
            
        # Log synthetic cache details
        for num_examples, data in self.synthetic_cache.items():
            logger.info(f"    {num_examples} corrected examples → {len(data)} synthetic examples")
    
    def clear_all_caches(self):
        """Clear all caches"""
        self.batch_analysis_cache.clear()
        self.final_summary_cache.clear()
        self.synthetic_cache.clear()
        
        # Remove cache files
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
    
    def _save_cache(self, cache_type: str):
        """Save specific cache to disk"""
        cache_file = self.cache_dir / f"{cache_type}_cache.pkl"
        
        if cache_type == 'batch_analysis':
            data = self.batch_analysis_cache
        elif cache_type == 'final_summary':
            data = self.final_summary_cache
        elif cache_type == 'synthetic':
            data = self.synthetic_cache
        else:
            return
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            # Don't crash if cache save fails
            print(f"Warning: Could not save {cache_type} cache: {e}")
    
    def _load_caches(self):
        """Load caches from disk"""
        cache_files = {
            'batch_analysis': self.cache_dir / "batch_analysis_cache.pkl",
            'final_summary': self.cache_dir / "final_summary_cache.pkl",
            'synthetic': self.cache_dir / "synthetic_cache.pkl"
        }
        
        for cache_type, cache_file in cache_files.items():
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    if cache_type == 'batch_analysis':
                        self.batch_analysis_cache = data
                    elif cache_type == 'final_summary':
                        self.final_summary_cache = data
                    elif cache_type == 'synthetic':
                        self.synthetic_cache = data
                        
                except Exception as e:
                    print(f"Warning: Could not load {cache_type} cache: {e}")


# Helper functions for backward compatibility with your original code
def get_batch_cache_key(batch_examples: List[Dict]) -> str:
    """Standalone function for backward compatibility"""
    cache_manager = CacheManager()
    return cache_manager.get_batch_cache_key(batch_examples)


def log_cache_status(cache_manager: CacheManager, logger: logging.Logger):
    """Standalone function for backward compatibility"""
    cache_manager.log_cache_status(logger)
