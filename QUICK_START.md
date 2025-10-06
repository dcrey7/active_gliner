# Quick Start Guide - Refactored LLM Abstractions

**Date:** 2025-10-06

---

## 🚀 Quick Start

### **1. Generate Labels (for Training)**

```python
from generation.label_generator import create_label_generator

# Create generator
generator = create_label_generator(
    backend_type='ollama',        # 'ollama', 'mistral', or 'cerebras'
    model_name='gemma3:12b',
    cache_type='disk',            # 'memory' or 'disk'
    cache_model_name='gemma3_12b'
)

# Generate labels
labels = generator.generate(
    low_n_examples=low_confidence_examples,
    num_samples=500,
    entity_types=['LOCATION', 'ACTOR', 'TITLE'],
    verbose=True
)
```

**Cache saves to:** `cache/labelling/gemma3_12b/gemma3_12b_500_labels.pkl`

---

### **2. Evaluate with LLM (for Testing)**

```python
from evaluation.ner_evaluator import create_ner_evaluator

# Create evaluator
evaluator = create_ner_evaluator(
    backend_type='cerebras',
    model_name='qwen-3-235b-a22b-instruct-2507',
    cache_type='disk',
    cache_model_name='qwen_3_235b'
)

# Generate predictions
predictions = evaluator.evaluate(
    test_data=test_examples,
    entity_types=['LOCATION', 'ACTOR', 'TITLE'],
    verbose=True
)
```

**Cache saves to:** `cache/evaluation/qwen_3_235b/qwen_3_235b_2500_evaluations.pkl`

---

### **3. Use Cerebras with Structured Output**

```python
from generation.label_generator import create_label_generator

# Structured output (JSON schema enforced by API)
generator = create_label_generator(
    backend_type='cerebras',
    model_name='qwen-3-235b-a22b-thinking-2507',  # Thinking model
    use_structured_output=True,                   # Enable structured output
    cache_type='disk'
)

labels = generator.generate(low_n, 500, entity_types)
```

**Benefit:** Faster, more reliable JSON (schema validation at API level)

---

## 📁 Cache Structure

Caches are saved in organized folders:

```
cache/
├── labelling/              # For label generation
│   ├── gemma3_12b/
│   │   ├── gemma3_12b_250_labels.pkl
│   │   └── gemma3_12b_500_labels.pkl
│   └── qwen_3_235b/
│       └── qwen_3_235b_500_labels.pkl
└── evaluation/             # For evaluation predictions
    └── gemma3_12b/
        └── gemma3_12b_2500_evaluations.pkl
```

---

## 🔧 Available Backends

### **Ollama** (Local)
```python
backend_type='ollama'
model_name='gemma3:12b'  # or 'llama3', 'mistral', etc.
```

### **Mistral Inference** (Local)
```python
backend_type='mistral'
model_path='/path/to/mistral/models/7B-Instruct-v0.3'  # optional
```

### **Cerebras API** (Cloud)
```python
backend_type='cerebras'
model_name='qwen-3-235b-a22b-instruct-2507'
# Requires CEREBRAS_API_KEY in .env
```

### **Cerebras Structured** (Cloud)
```python
backend_type='cerebras'
model_name='qwen-3-235b-a22b-thinking-2507'
use_structured_output=True
# Requires CEREBRAS_API_KEY in .env
```

---

## 🔑 Environment Variables

Create `.env` file in repository root:

```bash
# .env
CEREBRAS_API_KEY=your_api_key_here
```

The backends automatically load this file.

---

## 🧪 Testing

Test all abstractions:

```bash
cd test
uv run test_abstractions.py
```

Expected output:
```
✅ Phase 1: LLM Backend Layer - WORKING
✅ Phase 2: Prompt Building & Parsing - WORKING
✅ Phase 3: Caching - WORKING
```

---

## 📊 Validation Reports

The new validator provides detailed reports:

```python
from data.validator import NERValidator

validator = NERValidator(entity_types=['LOCATION', 'ACTOR'])
cleaned_data, report = validator.validate(ner_data, strict=True)

print(report.summary())
```

**Output:**
```
============================================================
VALIDATION REPORT
============================================================
Total examples processed: 100
Valid examples: 87
Removed examples: 13

Removal Details:
  • Out of bounds indices: 5 entities
    - Example 12: Entity (15, 20) but text length is 18
  • Invalid entity types: 3 entities
    Invalid types found: ['PERSON_NAME']
  • Invalid index order: 2 entities
============================================================
```

---

## 🔄 Switch Backends Easily

**Config-driven approach:**

```python
# config.py
LLM_CONFIG = {
    'backend': 'cerebras',  # Change this to switch
    'model': 'qwen-3-235b-a22b-instruct-2507',
    'cache': 'disk'
}

# main.py
from generation.label_generator import create_label_generator
from config import LLM_CONFIG

generator = create_label_generator(
    backend_type=LLM_CONFIG['backend'],
    model_name=LLM_CONFIG['model'],
    cache_type=LLM_CONFIG['cache']
)
```

**Switch to Ollama?** Just change config:
```python
LLM_CONFIG = {
    'backend': 'ollama',
    'model': 'gemma3:12b',
    'cache': 'disk'
}
```

---

## 📦 What Files to Use

### **For Label Generation (Training Data)**
```python
from generation.label_generator import create_label_generator
```

### **For Evaluation (Test Predictions)**
```python
from evaluation.ner_evaluator import create_ner_evaluator
```

### **For Validation (Data Quality)**
```python
from data.validator import NERValidator
```

### **Direct Backend Access (Advanced)**
```python
from llm_backends import BackendFactory

backend = BackendFactory.create('ollama', model_name='gemma3:12b')
text, in_tok, out_tok = backend.generate("Your prompt here")
```

---

## 🗂️ File Organization

**Old files (still exist, for reference):**
- `generation/gemma_labeler.py`
- `generation/mistral_labeler.py`
- `generation/api_labeler.py`
- `generation/enc_api_label.py`

**New files (use these):**
- `generation/label_generator.py` - Unified generator
- `evaluation/ner_evaluator.py` - Unified evaluator
- `data/validator.py` - Validator with reports

---

## 💡 Key Differences

### **Label Generator vs Evaluator**

**LabelGenerator** (for training):
- Strict validation (drops invalid examples)
- Used for creating training data
- Cache type: `"labelling"`

**NEREvaluator** (for evaluation):
- Preserves all indices (empty NER if invalid)
- Used for generating test predictions
- Cache type: `"evaluation"`

---

## 🎯 Next Steps

1. ✅ **Test the abstractions** - Run `test/test_abstractions.py`
2. ✅ **Try label generation** - Use `create_label_generator()`
3. ✅ **Try evaluation** - Use `create_ner_evaluator()`
4. ✅ **Check cache folder** - See `cache/labelling/` and `cache/evaluation/`
5. ✅ **Review validation reports** - See what was removed and why

---

## 📚 Full Documentation

For complete details, see:
- `updates/2025-10-06_complete_refactoring_summary.md` - Full refactoring summary
- `updates/2025-10-06_fixes_applied.md` - All fixes applied
- `llm_backends/README.md` - Backend documentation

---

**Happy Labeling! 🚀**
