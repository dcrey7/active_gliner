"""
Base Cache Interface
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class Cache(ABC):
    """Abstract base class for caching strategies"""

    @abstractmethod
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all cached items

        Returns:
            List of cached items
        """
        pass

    @abstractmethod
    def extend(self, items: List[Dict[str, Any]]) -> None:
        """
        Add multiple items to cache

        Args:
            items: List of items to cache
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached items"""
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Get cache size

        Returns:
            Number of items in cache
        """
        pass

    def get_subset(self, n: int) -> List[Dict[str, Any]]:
        """
        Get first n items from cache

        Args:
            n: Number of items to retrieve

        Returns:
            List of first n cached items
        """
        all_items = self.get_all()
        return all_items[:n]
