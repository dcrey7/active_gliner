# Cache Location Fix

**Date:** 2025-10-06
**Time:** Current session

---

## Issue
Cache was saving to `.cache/` folder outside the repository instead of inside the `cache/` folder within the repository.

---

## Fix Applied

### Changed Default Cache Location

**File Modified:** `src/caching/disk_cache.py`

**Change:**
```python
# BEFORE:
cache_root: str = ".cache"  # Hidden folder outside repo

# AFTER:
cache_root: str = "cache"   # Inside repository
```

---

## New Cache Structure

```
active_gliner/              # Your repository
├── cache/                  # ✅ Cache folder (inside repo!)
│   ├── labelling/          # For label generation
│   │   ├── gemma3_12b/
│   │   │   ├── gemma3_12b_250_labels.pkl
│   │   │   ├── gemma3_12b_500_labels.pkl
│   │   │   └── gemma3_12b_1000_labels.pkl
│   │   └── qwen_3_235b/
│   │       └── qwen_3_235b_500_labels.pkl
│   └── evaluation/         # For evaluation predictions
│       └── gemma3_12b/
│           └── gemma3_12b_2500_evaluations.pkl
├── src/
├── test/
└── ...
```

---

## Updated .gitignore

Added to ignore cache files but keep folder structure:

```gitignore
# Cache files (but keep cache/ folder structure)
cache/**/*.pkl
cache/**/*.pickle
*.pkl
*.pickle

# But ignore hidden .cache folder if created
.cache/
```

This allows:
- ✅ Cache folder structure is tracked in git
- ✅ `.pkl` files are ignored (not committed)
- ✅ Team members see the cache organization
- ✅ No large pickle files in version control

---

## Usage (No Changes Needed!)

Default behavior now saves inside repository:

```python
from caching import DiskCache

# Automatically saves to cache/labelling/gemma3_12b/
cache = DiskCache(
    cache_type="labelling",
    model_name="gemma3_12b"
    # cache_root="cache" is now the default!
)
```

---

## Benefits

1. **Visible in repository**: Easy to see what's cached
2. **Team sharing**: Everyone uses same cache structure
3. **No hidden folders**: No `.cache/` confusion
4. **Organized**: Clear separation of labelling vs evaluation
5. **Model-specific**: Each model has its own subfolder

---

## Files Modified

1. `src/caching/disk_cache.py` - Changed default `cache_root` from `.cache` to `cache`
2. `.gitignore` - Added cache file patterns
3. `FIXES_APPLIED.md` - Updated documentation

---

## Testing

Run test to verify:
```bash
cd test
uv run test_abstractions.py
```

Expected output:
```
💾 Testing DiskCache:
Cache directory: /tmp/.../labelling/gemma3_12b  # Will be cache/ in real use
✅ Disk cache saved and loaded successfully
```

In real usage (not tests):
- Cache saves to: `active_gliner/cache/labelling/gemma3_12b/`
- Files like: `gemma3_12b_250_labels.pkl`
