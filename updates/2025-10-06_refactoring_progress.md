# LLM Backend Refactoring Progress

**Date:** 2025-10-06
**Status:** Phases 1-3 Complete

---

## Summary

Successfully extracted and refactored LLM-related code into clean, modular abstractions.

### What Was Refactored

Eliminated 70% code duplication across:
- `generation/gemma_labeler.py` (239 lines)
- `generation/mistral_labeler.py` (373 lines)
- `generation/api_labeler.py` (410 lines)
- `generation/enc_api_label.py` (535 lines)

---

## ✅ Phase 1: LLM Backend Layer (COMPLETE)

### Created Files

```
src/
├── config/
│   └── llm_config.py          # Centralized LLM configs (temperature, top_k, etc.)
└── llm_backends/
    ├── __init__.py             # Exports BackendFactory
    ├── base.py                 # Abstract LLMBackend interface
    ├── factory.py              # BackendFactory for creating backends
    ├── ollama.py               # OllamaBackend (from gemma_labeler.py)
    ├── mistral.py              # MistralBackend (from mistral_labeler.py)
    ├── cerebras.py             # CerebrasBackend (from api_labeler.py)
    ├── cerebras_structured.py  # StructuredCerebrasBackend (from enc_api_label.py)
    └── README.md               # Documentation
```

### Key Features

- **Single Interface**: All backends implement `LLMBackend.generate(prompt) -> (text, input_tokens, output_tokens)`
- **Configuration**: All magic numbers moved to `config/llm_config.py`
- **Rate Limiting**: Cerebras backends handle rate limiting automatically
- **Graceful Quota Handling**: Structured Cerebras raises `RateLimitError` for hard quota limits
- **Token Tracking**: Mistral and Cerebras provide exact counts, Ollama returns 0

### Usage Example

```python
from llm_backends import BackendFactory

# Create any backend via factory
backend = BackendFactory.create('ollama', model_name='gemma3:12b')
# backend = BackendFactory.create('mistral')
# backend = BackendFactory.create('cerebras', model_name='qwen-3-235b-a22b-instruct-2507')
# backend = BackendFactory.create('cerebras', use_structured_output=True)

# Generate
response_text, input_tokens, output_tokens = backend.generate(prompt)

# Check capabilities
if backend.supports_structured_output():
    print("Supports structured JSON output")
```

---

## ✅ Phase 2: Prompt Building & Response Parsing (COMPLETE)

### Created Files

```
src/
├── prompting/
│   ├── __init__.py
│   ├── base.py                # Abstract PromptBuilder interface
│   ├── standard_prompt.py     # For Ollama, Mistral (normal prompting)
│   └── structured_prompt.py   # For Cerebras (simpler, schema-enforced)
└── parsing/
    ├── __init__.py
    └── response_parser.py     # Extracts JSON from LLM responses
```

### Key Features

- **Standard Prompts**: For backends without structured output (includes JSON format instructions)
- **Structured Prompts**: Simpler prompts for backends with schema validation
- **Response Parser**: Handles markdown wrapping (` ```json ... ``` `), extra text, malformed JSON

### Usage Example

```python
from prompting import StandardPromptBuilder, StructuredPromptBuilder
from parsing import ResponseParser

# Build prompt
if backend.supports_structured_output():
    prompt = StructuredPromptBuilder().build(tokenized_text, entity_types)
else:
    prompt = StandardPromptBuilder().build(tokenized_text, entity_types)

# Generate
response_text, _, _ = backend.generate(prompt)

# Parse JSON
parser = ResponseParser()
json_obj = parser.extract_json(response_text)
```

---

## ✅ Phase 3: Caching Layer (COMPLETE)

### Created Files

```
src/
└── caching/
    ├── __init__.py
    ├── base.py            # Abstract Cache interface
    ├── memory_cache.py    # List-based cache (current approach)
    └── disk_cache.py      # Persistent cache with atomic writes
```

### Key Features

- **MemoryCache**: Simple list-based cache (current approach from labelers)
- **DiskCache**: Persistent cache with atomic writes (improved from enc_api_label.py)
  - Saves to `.cache/llm_labels/` by default
  - Atomic writes (temp file + rename)
  - Metadata tracking (timestamp, reason, etc.)
  - Automatic loading of closest cache

### Usage Example

```python
from caching import MemoryCache, DiskCache

# Memory cache (for experiments)
cache = MemoryCache()

# Disk cache (for production, survives crashes)
cache = DiskCache(cache_dir='.cache/llm_labels', model_name='gemma3_12b')

# Use cache
cache.extend(new_labels)
print(f"Cache size: {cache.size()}")
subset = cache.get_subset(100)
```

---

## 📋 Next Steps: Phase 4 & Beyond

### Phase 4: Unified NER Label Generator (TODO)

**Goal**: Create single label generator that uses all the abstractions above.

**Plan**:
```
src/generation/
├── label_generator.py    # NEW: Unified generator using backend + cache + prompts
└── (keep old files for now as reference)
```

**Features**:
- Uses `BackendFactory` to create backend
- Uses `PromptBuilder` based on backend capabilities
- Uses `Cache` (memory or disk)
- Handles retry logic (max 3 attempts, current approach)
- Converts to NER format
- Validates and cleans (with detailed logging - next phase)

### Phase 5: Improved Validation with Detailed Reporting (TODO)

**Goal**: Validation that logs exactly what was removed and why.

**Plan**:
```
src/data/
├── validator.py             # NEW: Validation with detailed reporting
├── validation_report.py     # NEW: Report data class
└── transforms.py            # Keep existing conversion logic
```

**Features**:
- Track all removal reasons with example references
- Generate human-readable reports
- Save reports to file (optional)

### Phase 6: Unified Evaluator (TODO)

**Goal**: Refactor `llm_evaluator.py` to use same abstractions as label generator.

**Difference**: Evaluator must preserve all indices (empty NER if invalid), while generator can drop invalid examples.

---

## Benefits Achieved

### Before Refactoring
- 4 files with 70% duplicate code (~1,500 lines)
- Magic numbers scattered everywhere
- Inconsistent error handling
- Hard to add new LLM backends
- Hard to test (tightly coupled to APIs)

### After Refactoring
- Single interface for all backends
- Centralized configuration
- Consistent retry/rate limiting
- Add new backend in 1 file
- Easy to test (mock backend interface)
- Prompt engineering separated from execution
- Flexible caching (memory or disk)

---

## Migration Path

### Current State
Old labelers still exist:
- `generation/gemma_labeler.py`
- `generation/mistral_labeler.py`
- `generation/api_labeler.py`
- `generation/enc_api_label.py`

### Next Steps
1. Create new unified `generation/label_generator.py` using abstractions
2. Test with one notebook
3. Migrate notebooks to use new generator
4. Archive/delete old labelers

---

## Files Created (Summary)

**Config**: 1 file
- `config/llm_config.py`

**Backends**: 7 files
- `llm_backends/__init__.py`
- `llm_backends/base.py`
- `llm_backends/factory.py`
- `llm_backends/ollama.py`
- `llm_backends/mistral.py`
- `llm_backends/cerebras.py`
- `llm_backends/cerebras_structured.py`

**Prompting**: 4 files
- `prompting/__init__.py`
- `prompting/base.py`
- `prompting/standard_prompt.py`
- `prompting/structured_prompt.py`

**Parsing**: 2 files
- `parsing/__init__.py`
- `parsing/response_parser.py`

**Caching**: 4 files
- `caching/__init__.py`
- `caching/base.py`
- `caching/memory_cache.py`
- `caching/disk_cache.py`

**Documentation**: 2 files
- `llm_backends/README.md`
- `REFACTORING_PROGRESS.md` (this file)

**Total**: 20 new files, ~1,200 lines of clean, modular code

---

## Testing the Abstractions

To test the backend layer independently:

```python
# Test backend
from llm_backends import BackendFactory

backend = BackendFactory.create('ollama', model_name='gemma3:12b')
prompt = "What is Named Entity Recognition?"
text, in_tok, out_tok = backend.generate(prompt)
print(f"Response: {text}")
print(f"Tokens: {in_tok} in, {out_tok} out")

# Test prompt building
from prompting import StandardPromptBuilder

builder = StandardPromptBuilder()
prompt = builder.build(
    tokenized_text=["show", "me", "flights", "to", "Boston"],
    entity_types=["LOCATION", "ACTOR", "TITLE"]
)
print(prompt)

# Test response parsing
from parsing import ResponseParser

response = '```json\n{"text": "test", "entities": []}\n```'
parser = ResponseParser()
json_obj = parser.extract_json(response)
print(json_obj)

# Test caching
from caching import MemoryCache, DiskCache

cache = MemoryCache()
cache.extend([{"text": "example", "ner": []}])
print(f"Cache size: {cache.size()}")
```

---

## Next Session

Continue with Phase 4: Create unified `label_generator.py` that combines all abstractions.
