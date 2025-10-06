"""
Memory Cache Implementation
Simple list-based cache (current approach from gemma_labeler.py, etc.)
"""

from typing import List, Dict, Any
from .base import Cache


class MemoryCache(Cache):
    """In-memory cache using list (current approach)"""

    def __init__(self):
        """Initialize empty memory cache"""
        self._cache: List[Dict[str, Any]] = []

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all cached items

        Returns:
            List of all cached items
        """
        return self._cache

    def extend(self, items: List[Dict[str, Any]]) -> None:
        """
        Add multiple items to cache

        Args:
            items: List of items to cache
        """
        self._cache.extend(items)

    def clear(self) -> None:
        """Clear all cached items"""
        self._cache.clear()

    def size(self) -> int:
        """
        Get cache size

        Returns:
            Number of items in cache
        """
        return len(self._cache)
