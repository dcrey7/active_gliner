# Active GLiNER

Information extraction from text using GLiNER and LLM distillation. This package enables cost-effective, locally-runnable named entity recognition by distilling LLM-extracted labels into lightweight GLiNER models.

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

## Installation

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

## Quick Start

### 1. Zero-Shot Prediction

```python
from main import zeroshot
from data.loader import load_mit_dataset

# Load data
test_data, entity_types = load_mit_dataset("data/mit-movie/test.json", "data/mit-movie/labels.json")

# Run zero-shot prediction
predictions = zeroshot(
    data=test_data,
    entity_types=entity_types,
    threshold=0.5
)
```

### 2. Rank Examples by Uncertainty

```python
from main import ranking

# Find most uncertain examples for active learning
uncertain_examples = ranking(
    predictions=predictions,
    data=test_data,
    entity_types=entity_types,
    n_examples=100,
    strategy='mse'  # or 'min_score'
)
```

### 3. Fine-Tune with Ground Truth

```python
from main import finetune

# Fine-tune with pure ground truth
adapter = finetune(
    training_data=train_data,
    eval_data=test_data,
    entity_types=entity_types,
    adapter_save_path='./models/adapter_gt'
)
```

### 4. Fine-Tune with LLM + Ground Truth Mix

```python
# Fine-tune with 50% LLM labels + 50% ground truth
adapter = finetune(
    training_data=train_data,
    eval_data=test_data,
    entity_types=entity_types,
    adapter_save_path='./models/adapter_mixed',
    llm_backend='ollama',
    llm_model='gemma3:12b',
    mix_ratio=50  # 50% GT, 50% LLM
)
```

### 5. Predict with Fine-Tuned Model

```python
from main import predict

predictions = predict(
    data=new_data,
    entity_types=entity_types,
    adapter_path='./models/adapter_gt'
)
```

### 6. Evaluate Results

```python
from main import evaluate

results = evaluate(
    data=test_data,
    entity_types=entity_types,
    predictions=predictions,
    model_type='gloner',
    has_ground_truth=True
)

print(f"F1 Score: {results['overall_metrics']['overall_f1_pct']:.2f}%")
```

## Repository Structure

```
active_gliner/
├── src/                          # Source code
│   ├── main.py                   # Entry point API (5 main functions)
│   ├── models/
│   │   └── gloner.py             # GLiNER + LoRA wrapper
│   ├── generation/
│   │   └── llm_inference.py      # LLM label generation
│   ├── evaluation/
│   │   ├── eval.py               # Evaluation functions
│   │   ├── eval_metrics.py       # Metrics calculation
│   │   └── helper.py             # Visualization helpers
│   ├── training/
│   │   └── trainer.py            # LoRA training with monitoring
│   ├── data/
│   │   ├── loader.py             # Data loading utilities
│   │   ├── transforms.py         # Data transformations
│   │   └── validator.py          # NER data validation
│   ├── llm_backends/
│   │   ├── ollama.py             # Ollama backend
│   │   ├── cerebras.py           # Cerebras backend
│   │   └── mistral.py            # Mistral backend
│   ├── prompting/
│   │   ├── base.py               # Prompt builder base
│   │   ├── standard_prompt.py    # Standard prompting
│   │   └── structured_prompt.py  # Structured output prompting
│   ├── parsing/
│   │   └── response_parser.py    # LLM response parsing
│   ├── caching/
│   │   ├── disk_cache.py         # Disk-based caching
│   │   └── memory_cache.py       # In-memory caching
│   ├── selection/
│   │   └── strategies.py         # Active learning selection
│   ├── config/
│   │   ├── settings.py           # Project settings
│   │   ├── lora_defaults.py      # LoRA configuration
│   │   ├── training_config.py    # Training parameters
│   │   └── llm_config.py         # LLM backend configs
│   └── utils/
│       ├── logging.py            # Logging utilities
│       ├── device.py             # GPU/CPU setup
│       ├── memory.py             # Memory management
│       └── reproducibility.py    # Seed setting
├── test/                         # Experiment scripts
│   ├── test_package_api.py       # API demonstration
│   ├── test_gloner.py            # GLONER model tests
│   ├── test_labeling.py          # LLM labeling tests
│   ├── test_llm_evaluate.py      # LLM evaluation tests
│   ├── confidence_test_base.py   # Base model comparison
│   ├── confidence_test_FT.py     # Fine-tuning comparison
│   ├── mixed_test_FT.py          # Mixed ratio experiments
│   └── mixed_api.py              # API quota handling tests
├── notebooks/                    # Research notebooks (legacy)
├── data/                         # Datasets
│   └── mit-movie/                # MIT Movie dataset
├── results/                      # Experiment results
├── models/                       # Saved adapters
├── logs/                         # Training logs
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Main API Functions

### `zeroshot()`
Zero-shot prediction with GLiNER (optionally with LoRA initialization)

**Use case:** Quick predictions without training

### `ranking()`
Rank examples by uncertainty for active learning selection

**Use case:** Find most uncertain examples to send for annotation

### `finetune()`
Fine-tune GLiNER with LoRA on GT + LLM mixed labels

**Use case:** Train custom models with business annotations + LLM distillation

### `predict()`
Predict with fine-tuned adapter

**Use case:** Production inference with trained models

### `evaluate()`
Evaluate predictions or models

**Use case:** Measure performance on test sets

## Supported LLM Backends

- **Ollama** - Local LLM inference (e.g., Gemma, Llama)
- **Cerebras** - Fast cloud inference with structured outputs
- **Mistral** - Mistral AI API

## Workflow

1. **Zero-shot baseline** - Run GLiNER without training to establish baseline
2. **Active learning** - Rank examples by uncertainty to find hard cases
3. **LLM labeling** - Generate training labels using LLMs
4. **Mix with GT** - Combine LLM labels with business ground truth
5. **Fine-tune** - Train GLiNER with LoRA on mixed labels
6. **Evaluate** - Compare fine-tuned model vs baseline vs LLM
7. **Deploy** - Use fine-tuned adapter for production

## Experiments

Run experiments from the `test/` directory:

```bash
# Test package API
python test/test_package_api.py

# Base model comparison
python test/confidence_test_base.py

# Fine-tuning comparison
python test/confidence_test_FT.py

# Mixed ratio experiments
python test/mixed_test_FT.py
```

## Key Features

- **LoRA Fine-Tuning** - Efficient training with minimal parameters
- **LLM Distillation** - Transfer LLM knowledge to lightweight models
- **Mixed Training** - Combine GT annotations with LLM labels
- **Active Learning** - Uncertainty-based example selection
- **Disk Caching** - Persistent LLM label caching
- **Multi-Backend** - Support for Ollama, Cerebras, Mistral
- **Comprehensive Evaluation** - Entity-level and example-level metrics

## Configuration

### LoRA Configuration (`config/lora_defaults.py`)
- Rank: 32
- Alpha: 64
- Target modules: Attention layers, span representation layers

### Training Configuration (`config/training_config.py`)
- Steps: 1000
- Learning rate: 2.1e-4
- Warmup ratio: 0.07
- Early stopping: 3 patience

### LLM Configuration (`config/llm_config.py`)
- Ollama: Top-p 0.8, Temperature 0.3
- Cerebras: Max tokens 60k, structured output support
- Mistral: Max tokens 50k, Temperature 0.3

## Results

Results from experiments are saved to `results/` directory:
- CSV files with metrics
- PNG plots for visualization
- JSON files with predictions and labels

## Development

The codebase follows SOLID principles:
- **Single Responsibility** - Each module has one clear purpose
- **Open/Closed** - Extensible via backends, strategies, prompts
- **Liskov Substitution** - Abstract base classes for backends/caches
- **Interface Segregation** - Small, focused interfaces
- **Dependency Inversion** - Factory patterns for backend selection

## License

MIT License

## Citation

If you use this code, please cite:
```
@software{active_gliner,
  title = {Active GLiNER: LLM Distillation for Information Extraction},
  year = {2025},
  url = {https://github.com/yourusername/active_gliner}
}
```
