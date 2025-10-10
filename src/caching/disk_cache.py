"""
Disk Cache Implementation
Persistent cache with organized folder structure and pickle support
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import Cache
from utils.logging import get_logger


class DiskCache(Cache):
    """
    Persistent disk-based cache with organized structure

    Cache Structure (inside repository):
    cache/
    ├── labelling/              # For label generation
    │   ├── gemma3_12b/
    │   │   ├── gemma3_12b_250_labels.pkl
    │   │   ├── gemma3_12b_500_labels.pkl
    │   │   └── gemma3_12b_1000_labels.pkl
    │   └── qwen_3_235b/
    │       └── qwen_3_235b_500_labels.pkl
    └── evaluation/             # For evaluation predictions
        └── gemma3_12b/
            └── gemma3_12b_2500_evaluations.pkl
    """

    def __init__(
        self,
        cache_type: str = "labelling",  # "labelling" or "evaluation"
        model_name: str = "default",
        cache_root: str = None
    ):
        """
        Initialize disk cache with organized structure

        Args:
            cache_type: Type of cache ("labelling" or "evaluation")
            model_name: Model name for folder and file naming
            cache_root: Root cache directory (default: None - uses project root/cache)
        """
        self.cache_type = cache_type
        self.model_name = model_name.replace("/", "_").replace(":", "_")

        # If cache_root not provided, use project root/cache
        if cache_root is None:
            # Find project root (where pyproject.toml or .git exists)
            current = Path(__file__).resolve()
            for parent in current.parents:
                if (parent / 'pyproject.toml').exists() or (parent / '.git').exists():
                    self.cache_root = parent / "cache"
                    break
            else:
                # Fallback to relative path if project root not found
                self.cache_root = Path("cache")
        else:
            self.cache_root = Path(cache_root)

        # Create organized structure: cache/labelling/gemma3_12b/
        self.cache_dir = self.cache_root / cache_type / self.model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger("DiskCache")
        self.logger.info(f"Cache directory: {self.cache_dir}")

        self._cache: List[Dict[str, Any]] = []
        self._loaded = False

    def _get_cache_filename(self, num_labels: int) -> Path:
        """
        Generate cache filename: model_name_num_labels.pkl

        Example: gemma3_12b_250_labels.pkl

        Args:
            num_labels: Number of labels

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{self.model_name}_{num_labels}_labels.pkl"

    def _load_from_disk(self, target_labels: int) -> bool:
        """
        Load cache from disk if available

        Args:
            target_labels: Target number of labels to load

        Returns:
            True if cache was loaded, False otherwise
        """
        if self._loaded:
            return True

        # Try exact match first
        cache_file = self._get_cache_filename(target_labels)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self._cache = cache_data.get('labels', [])
                self.logger.info(f"✅ Loaded {len(self._cache)} labels from: {cache_file.relative_to(self.cache_root)}")
                self._loaded = True
                return True
            except Exception as e:
                self.logger.warning(f"Failed to load cache from {cache_file.name}: {e}")

        # Find the largest existing cache file that's <= target_labels
        cache_files = list(self.cache_dir.glob(f"{self.model_name}_*_labels.pkl"))

        # Sort by the number in filename (descending)
        def get_num_labels(path):
            try:
                return int(path.stem.split('_')[-2])
            except:
                return 0

        cache_files = sorted(cache_files, key=get_num_labels, reverse=True)

        for cache_file in cache_files:
            # Extract number from filename: gemma3_12b_10_labels.pkl -> 10
            try:
                num_in_file = int(cache_file.stem.split('_')[-2])
                if num_in_file <= target_labels:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    self._cache = cache_data.get('labels', [])
                    self.logger.info(f"✅ Loaded {len(self._cache)} labels from: {cache_file.relative_to(self.cache_root)}")
                    self._loaded = True
                    return True
            except Exception as e:
                self.logger.warning(f"Failed to load cache from {cache_file.name}: {e}")
                continue

        self.logger.info("📝 No existing cache found, starting fresh")
        self._loaded = True
        return False

    def save_to_disk(self, reason: str = "completed") -> None:
        """
        Save cache to disk atomically using pickle

        Args:
            reason: Reason for saving (for metadata)
        """
        if not self._cache:
            self.logger.warning("No labels to save to cache")
            return

        num_labels = len(self._cache)
        cache_file = self._get_cache_filename(num_labels)
        temp_file = cache_file.with_suffix('.tmp')

        try:
            cache_data = {
                "metadata": {
                    "cache_type": self.cache_type,
                    "model_name": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "total_labels": num_labels,
                    "reason": reason
                },
                "labels": self._cache
            }

            with open(temp_file, 'wb') as f:
                pickle.dump(cache_data, f)

            temp_file.rename(cache_file)
            self.logger.info(f"💾 Saved {num_labels} labels to: {cache_file.relative_to(self.cache_root)} (reason: {reason})")

        except Exception as e:
            self.logger.error(f"Failed to save cache to {cache_file.name}: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all cached items (loads from disk if not loaded)

        Returns:
            List of all cached items
        """
        if not self._loaded:
            # Try to load with a large target (will find any existing cache)
            self._load_from_disk(target_labels=100000)

        return self._cache

    def extend(self, items: List[Dict[str, Any]]) -> None:
        """
        Add multiple items to cache and save to disk

        Args:
            items: List of items to cache
        """
        if not self._loaded:
            self.get_all()  # Load existing cache first

        self._cache.extend(items)
        self.save_to_disk(reason="extended")

    def clear(self) -> None:
        """Clear cache (both memory and disk)"""
        self._cache.clear()
        self._loaded = False

    def size(self) -> int:
        """
        Get cache size

        Returns:
            Number of items in cache
        """
        if not self._loaded:
            self.get_all()

        return len(self._cache)

    def load_or_create(self, target_labels: int) -> None:
        """
        Explicitly load cache for target number of labels

        Args:
            target_labels: Target number of labels to load
        """
        self._load_from_disk(target_labels)

    def list_cached_files(self) -> List[str]:
        """
        List all cache files in this model's directory

        Returns:
            List of cache filenames
        """
        cache_files = sorted(self.cache_dir.glob(f"{self.model_name}_*_labels.pkl"))
        return [f.name for f in cache_files]
