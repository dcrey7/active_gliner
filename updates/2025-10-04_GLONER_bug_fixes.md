# GLONER Bug Fixes

**Date:** 2025-10-04 (Post-Implementation)
**Status:** ✅ Fixed

## Issues Fixed

### Issue 1: Removed Incorrect `model.train()` Call

**Problem:**
- `for_training()` method was calling `self.model.train()`
- This only sets PyTorch to training mode (enables dropout, batch norm updates)
- This is NOT how GLiNER training works!

**What Actually Trains:**
- GLiNER uses HuggingFace `Trainer` class
- Training happens when you call `trainer.train()` in `train_lora_model()`
- The `model.train()` call was misleading and unnecessary

**Fix:**
```python
# BEFORE (WRONG):
def for_training(self):
    self._load_base_model()
    self._apply_lora()
    self.model.train()  # ❌ This doesn't train!
    return self

# AFTER (CORRECT):
def for_training(self):
    self._load_base_model()
    self._apply_lora()
    # No model.train() - trainer handles it!
    return self
```

**Note:** `model.eval()` is kept in `for_inference()` - this is correct as it sets evaluation mode.

---

### Issue 2: Made `max_length` Customizable

**Problem:**
- `max_length` was hardcoded to 8192 in `_load_base_model()`
- Different GLiNER models may need different maximum sequence lengths
- No way to customize it

**Fix:**

#### Updated `config/lora_defaults.py`:
```python
DEFAULT_GLINER_MODEL = "knowledgator/modern-gliner-bi-large-v1.0"
DEFAULT_MAX_LENGTH = 8192  # ← Added this
```

#### Updated GLONER constructor:
```python
def __init__(self, model_name: str, lora_config: dict, max_length: int, logger=None):
    self.model_name = model_name
    self.lora_config = lora_config
    self.max_length = max_length  # ← Now a parameter
    self.logger = logger
    self.model = None
```

#### Updated factory methods:
```python
@classmethod
def default(cls, logger):
    return cls(DEFAULT_GLINER_MODEL, DEFAULT_LORA_CONFIG.copy(), DEFAULT_MAX_LENGTH, logger)

@classmethod
def custom(cls, logger, model_name=None, max_length=None, **lora_params):
    model_name = model_name or DEFAULT_GLINER_MODEL
    max_length = max_length or DEFAULT_MAX_LENGTH  # ← Can override
    lora_config = DEFAULT_LORA_CONFIG.copy()
    lora_config.update(lora_params)
    return cls(model_name, lora_config, max_length, logger)
```

#### Updated `_load_base_model()`:
```python
def _load_base_model(self):
    self.model = GLiNER.from_pretrained(self.model_name)
    self.model.config.max_len = self.max_length  # ← Uses parameter

    if hasattr(self.model.data_processor, 'transformer_tokenizer'):
        self.model.data_processor.transformer_tokenizer.model_max_length = self.max_length  # ← Uses parameter
```

---

## Usage After Fixes

### Default max_length (8192):
```python
gloner = GLONER.default(logger).for_training()
```

### Custom max_length:
```python
gloner = GLONER.custom(logger, max_length=4096).for_training()
```

### Custom model with custom max_length:
```python
gloner = GLONER.custom(
    logger,
    model_name="knowledgator/gliner-base",
    max_length=4096,
    r=16
).for_training()
```

---

## Files Modified

1. `src/config/lora_defaults.py` - Added `DEFAULT_MAX_LENGTH`
2. `src/models/gloner.py` -
   - Added `max_length` parameter to constructor
   - Updated `.default()` factory method
   - Updated `.custom()` factory method with `max_length` parameter
   - Removed `model.train()` from `for_training()`
   - Updated `_load_base_model()` to use `self.max_length`

---

## Why These Fixes Matter

### Issue 1 (model.train()):
- **Misleading**: Looked like it was training, but wasn't
- **Unnecessary**: Trainer sets this automatically
- **Cleaner**: Now it's clear that `for_training()` just prepares the model

### Issue 2 (max_length):
- **Flexibility**: Different models need different lengths
- **Correctness**: Some models may fail with wrong max_length
- **Future-proof**: Easy to support new models with different requirements

---

## Testing

Both default and custom configurations still work:
```python
# Test 1: Default (max_length=8192)
gloner = GLONER.default(logger).for_training()
assert gloner.max_length == 8192

# Test 2: Custom max_length
gloner = GLONER.custom(logger, max_length=4096).for_training()
assert gloner.max_length == 4096
```

---

## Summary

✅ Removed misleading `model.train()` call
✅ Made `max_length` customizable for both default and custom models
✅ All changes backward compatible (default behavior unchanged)
✅ Clearer code that matches actual GLiNER training flow
