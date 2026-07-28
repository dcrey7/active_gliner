# Active GLiNER

Information extraction from text using GLiNER and LLM distillation. This package enables cost-effective, locally-runnable named entity recognition by distilling LLM-extracted labels into lightweight GLiNER models. For the full detailed report please check https://drive.google.com/file/d/1eo1z6MbX-gSsD8jMPwdCOveldRVPxqrF/view?usp=drive_link

## Overview

LLMs are powerful for information extraction but suffer from:
- High inference costs
- Cannot be fine-tuned easily
- Too large to run locally

Active GLiNER solves this by:
1. Using LLMs to generate high-quality training labels
2. Distilling those labels into fine-tuned GLiNER models
3. Mixing LLM labels with ground truth from business annotations
4. Producing lightweight, locally-runnable models with near-LLM performance

## Using Active GLiNER Library

### Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### Python API

```python
# Import from src/active_gliner
import sys
sys.path.insert(0, '/app/src')

from active_gliner.get_model.DefaultModel import DefaultModel
from active_gliner.selection.strategy import calculate_mse_score
from active_gliner.llm.backends.cerebras import CerebrasBackend
```

### Example: Zero-Shot Prediction

```python
from active_gliner.get_model.DefaultModel import DefaultModel

# Initialize and load model for inference
model = DefaultModel()
model.load_for_inference()  # Loads base GLiNER model 

# Predict entities
predictions = model.predict_entities(
    text="John works at OpenAI in San Francisco",
    entity_types=["person", "organization", "location"],
    threshold=0.5,
    flat_ner=False
)
```

### Example: Prediction with Fine-Tuned Model

```python
from active_gliner.get_model.DefaultModel import DefaultModel

# Initialize and load model with adapter
model = DefaultModel()
model.load_for_inference(adapter_path='./models/my_adapter')

# Predict entities
predictions = model.predict_entities(
    text="John works at OpenAI in San Francisco",
    entity_types=["person", "organization", "location"],
    threshold=0.5,
    flat_ner=False
)
```

### Example: Fine-Tuning

```python
from active_gliner.get_model.DefaultModel import DefaultModel
from active_gliner.create_data.gliner_format import convert_raw_json_to_gliner_training

# Load training data in GLiNER format
# Format: [{"tokenized_text": [...], "ner": [[start, end, "label"], ...]}, ...]
train_data = convert_raw_json_to_gliner_training(raw_train_data)
eval_data = convert_raw_json_to_gliner_training(raw_eval_data)

# Initialize model and load for training
model = DefaultModel()
model.load_for_training()  # Loads base model with LoRA config

# Fine-tune the model
model.fit(
    train_data=train_data,
    eval_data=eval_data,
    adapter_save_path='./models/my_adapter'
)

# After training, load for inference
model = DefaultModel()
model.load_for_inference(adapter_path='./models/my_adapter')
```

### Example: LLM Label Generation

```python
from active_gliner.llm.backends.cerebras import CerebrasBackend
from active_gliner.llm.prompts import StandardPrompt

# Initialize backend
backend = CerebrasBackend(api_key="your_key")

# Generate labels
prompt = StandardPrompt(entity_types=["person", "organization"])
response = backend.generate(
    text="John works at OpenAI",
    prompt=prompt
)
```

---

## Experiments

All experiments are located in `src/active_gliner/run_experiments/` and can be configured via their respective config sections.

### 1. Active Learning Strategies
**File:** `exp_active_learning_confidence_strategies.py`

Compares different selection strategies for active learning:
- MSE (Mean Squared Error)
- Least Confidence
- MNLP (Mean Negative Log Probability)
- Random baseline

**Run:**
```bash
python src/active_gliner/run_experiments/exp_active_learning_confidence_strategies.py
```

**Configuration:** Edit experiment config at the top of the file to change dataset, model, or strategies.

**Output:** `results/exp_active_learning_confidence_strategies/`

---

### 2. GLiNER vs LLM Baseline
**File:** `exp_confidence_gliner_llm_baseline_f1.py`

Compares zero-shot performance between GLiNER and LLM backends.

**Run:**
```bash
python src/active_gliner/run_experiments/exp_confidence_gliner_llm_baseline_f1.py
```

**Configuration:** Swap LLM backend and model in experiment config.

**Output:** `results/exp_confidence_gliner_llm_baseline_f1/`

---

### 3. Fine-Tuning Comparison
**File:** `exp_confidence_gliner_llm_ft_f1.py`

Compares fine-tuned GLiNER on:
- Ground truth labels only
- LLM-generated labels only

**Run:**
```bash
python src/active_gliner/run_experiments/exp_confidence_gliner_llm_ft_f1.py
```

**Configuration:** Change training data source in experiment config.

**Output:** `results/exp_confidence_gliner_llm_ft_f1/`

---

### 4. Mixed Training Ratios
**File:** `exp_confidence_mixed_ft_f1.py`

Tests different mixing ratios of ground truth and LLM labels (0%, 25%, 50%, 75%, 100%).

**Run:**
```bash
python src/active_gliner/run_experiments/exp_confidence_mixed_ft_f1.py
```

**Configuration:** Adjust mixing ratios in experiment config.

**Output:** `results/exp_confidence_mixed_ft_f1/`

---

### 5. Hyperparameter Optimization
**File:** `exp_gliner_best_hyperparameters.py`

Uses Optuna to find optimal training hyperparameters (learning rate, LoRA rank/alpha, batch size, etc.).

**Run:**
```bash
python src/active_gliner/run_experiments/exp_gliner_best_hyperparameters.py
```

**Configuration:** Set number of trials and parameter search space in experiment config.

**Output:** `results/exp_gliner_best_hyperparameters/`

---

### 6. LoRA Layer Configuration
**File:** `exp_gliner_best_lora_layers.py`

Tests different LoRA target module configurations to find optimal layers for fine-tuning.

**Run:**
```bash
python src/active_gliner/run_experiments/exp_gliner_best_lora_layers.py
```

**Configuration:** Define target layer combinations in experiment config.

**Output:** `results/exp_gliner_best_lora_layers/`

---

### 7. Threshold Optimization
**File:** `exp_gliner_llm_threshold_f1.py`

Optimizes confidence thresholds for GLiNER and LLM predictions.

**Run:**
```bash
python src/active_gliner/run_experiments/exp_gliner_llm_threshold_f1.py
```

**Configuration:** Set threshold range in experiment config.

**Output:** `results/exp_gliner_llm_threshold_f1/`

---

### Experiment Results

All experiments save results to `results/exp_*/`:
- `plots/` - Visualization plots (PNG)
- `csv/` - Metrics and results (CSV)
- `logs/` - Training logs
- `adapters/` - Fine-tuned model adapters

---

## Environment Configuration

Create a `.env` file in the project root:

```bash
LABEL_STUDIO_API_KEY=your_labelstudio_api_key_here
CEREBRAS_API_KEY=your_cerebras_api_key_here
```

---

## Testing

```bash
# Test core functionality
python test/test_base.py
python test/test_defaultmodel.py
python test/test_evaluate.py

# Test LLM integration
python test/test_llm_labels.py

# Test selection strategies
python test/test_selection.py
```

---

