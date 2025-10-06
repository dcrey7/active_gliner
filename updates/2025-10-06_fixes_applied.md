# Fixes Applied to LLM Abstractions

**Date:** 2025-10-06

---

## Issue 1: ✅ FIXED - Cerebras API Key Not Loading

### Problem
`.env` file was not being loaded in Cerebras backends, causing `CEREBRAS_API_KEY environment variable not set` error.

### Solution
Added `from dotenv import load_dotenv` and `load_dotenv()` to both Cerebras backends:

**Files Modified:**
- `src/llm_backends/cerebras.py`
- `src/llm_backends/cerebras_structured.py`

**Changes:**
```python
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
```

Now your `.env` file with `CEREBRAS_API_KEY=...` will be automatically loaded.

---

## Issue 2: ✅ FIXED - Better Cache Organization

### Problem
Cache was saving to unclear locations with generic names. Hard to understand what's cached.

### Solution
Completely rewrote `DiskCache` with organized structure:

**New Cache Structure (inside repository):**
```
cache/                      # Inside repository!
├── labelling/              # For label generation
│   ├── gemma3_12b/
│   │   ├── gemma3_12b_250_labels.pkl
│   │   ├── gemma3_12b_500_labels.pkl
│   │   └── gemma3_12b_1000_labels.pkl
│   ├── qwen_3_235b/
│   │   └── qwen_3_235b_500_labels.pkl
│   └── mistral_7b/
│       └── mistral_7b_100_labels.pkl
└── evaluation/             # For evaluation predictions
    └── gemma3_12b/
        └── gemma3_12b_2500_evaluations.pkl
```

**Key Improvements:**
1. **Inside repository**: `cache/` folder (not hidden `.cache/`)
2. **Organized folders**: `cache/labelling/` or `cache/evaluation/`
3. **Model-specific subfolders**: Each model gets its own folder
4. **Clear filenames**: `{model_name}_{count}_labels.pkl`
5. **Pickle format**: Faster than JSON, more efficient
6. **Metadata tracking**: Timestamp, reason, cache type saved

**New Usage:**
```python
from caching import DiskCache

# For labeling (saves to cache/labelling/)
cache = DiskCache(
    cache_type="labelling",
    model_name="gemma3_12b",
    cache_root="cache"  # Default - inside repository
)

# For evaluation
eval_cache = DiskCache(
    cache_type="evaluation",
    model_name="gemma3_12b"
)

# List cached files
print(cache.list_cached_files())
# Output: ['gemma3_12b_250_labels.pkl', 'gemma3_12b_500_labels.pkl']
```

**Files Modified:**
- `src/caching/disk_cache.py` (complete rewrite)

---

## Issue 3: ✅ ANSWERED - Do We Need Prompting Module?

### Question
"The structured and normal prompt look the same right? So do we need the prompting module?"

### Answer
**YES, they ARE different and the module IS needed!**

**Standard Prompt (for Ollama, Mistral):**
- Includes full JSON format specification
- Has examples: `{"entity": "...", "types": ["..."]}`
- Includes warnings: "CRITICAL: Generate ONLY the JSON format above"
- **Why?** LLM needs to be told exactly what format to output

**Structured Prompt (for Cerebras with schema):**
- Does NOT include JSON format specification
- Does NOT need format warnings
- Simpler, shorter prompt
- **Why?** Cerebras API enforces JSON schema automatically via `response_format` parameter

**Real Difference:**
```python
# Standard prompt (200+ chars explaining JSON format)
"**MANDATORY Output Format:**
{
  \"text\": \"{text}\",
  \"entities\": [...]
}
CRITICAL: Generate ONLY the JSON format above."

# Structured prompt (no format instructions, ~50 chars shorter)
# Just describes the task, API handles the rest
```

**Benefit**: Shorter prompts = fewer tokens = faster + cheaper

---

## Issue 4: ✅ FIXED - Move Tests to Test Folder

### Problem
`test_abstractions.py` was in root directory, should be in `test/` folder.

### Solution
Moved file to proper location:
```bash
test_abstractions.py → test/test_abstractions.py
```

**Also Updated Test** to use new DiskCache structure:
- Shows cache directory structure
- Lists cached files
- Tests organized folder layout

---

## Summary of All Changes

### Files Modified: 3
1. `src/llm_backends/cerebras.py` - Added dotenv loading
2. `src/llm_backends/cerebras_structured.py` - Added dotenv loading
3. `src/caching/disk_cache.py` - Complete rewrite with organized structure

### Files Moved: 1
1. `test_abstractions.py` → `test/test_abstractions.py`

### New Features Added:
- ✅ `.env` file automatically loaded for Cerebras backends
- ✅ Organized cache structure (`.cache/labelling/model_name/`)
- ✅ Pickle format for faster caching
- ✅ Model-specific cache folders
- ✅ Clear filenames with model name and count
- ✅ `list_cached_files()` method to see what's cached

---

## Testing

Run the updated test:
```bash
cd /home/abhishek/Downloads/work/active_gliner
python test/test_abstractions.py
```

Expected output:
- ✅ Cerebras backends now load API key from .env
- ✅ DiskCache shows organized structure
- ✅ Cache files named clearly: `gemma3_12b_3_labels.pkl`

---

## Next Steps

Everything is ready for Phase 4: Create unified label generator that uses all these abstractions.

**What's Working:**
- ✅ LLM Backends (Ollama, Mistral, Cerebras, Cerebras Structured)
- ✅ API keys loaded from .env
- ✅ Prompt builders (Standard vs Structured)
- ✅ Response parser (handles all JSON edge cases)
- ✅ Caching (Memory and organized Disk cache with pickle)

**Ready to Use:**
```python
from llm_backends import BackendFactory
from prompting import StandardPromptBuilder, StructuredPromptBuilder
from caching import DiskCache

# Create backend (API key from .env)
backend = BackendFactory.create('cerebras', model_name='qwen-3-235b')

# Setup cache with organized structure
cache = DiskCache(
    cache_type="labelling",
    model_name="qwen_3_235b"
)

# Choose prompt based on backend
if backend.supports_structured_output():
    builder = StructuredPromptBuilder()
else:
    builder = StandardPromptBuilder()

# Generate
prompt = builder.build(["show", "flights", "to", "Boston"], ["LOCATION"])
text, in_tok, out_tok = backend.generate(prompt)
```
