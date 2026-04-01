# Literature Review: GLiNER + LoRA + Active Learning Research
**Date:** 2026-04-01, ~12:00 PM

---

## 1. GLiNER Current State of the Art

### Latest Version
- **GLiNER library (PyPI):** v0.2.26 (released March 19, 2026), with 39 total releases
- **Original paper:** "GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer" (arxiv 2311.08526), published at NAACL 2024
- **Best model variants available:**
  - `urchade/gliner_medium-v2.1` (original uni-encoder)
  - `urchade/gliner_large-v2.1`
  - `knowledgator/modern-gliner-bi-large-v1.0` (ModernBERT-large + BGE-base-en bi-encoder)
  - `knowledgator/gliner-multitask-large-v0.5` and `v1.0`
  - `knowledgator/gliner-bi-large-v1.0` and `v2.0`
  - `knowledgator/gliner-poly-base-v1.0` (poly-encoder)
  - `knowledgator/gliner-bi-llama-v1.0` (Llama-based)

### Architecture Evolution
1. **Uni-encoder (original):** Single bidirectional transformer encodes both text and entity labels together. Simple but bottlenecked by context window when many entity types are present.
2. **Bi-encoder (2024):** Separates text encoder and entity label encoder. Up to 130x throughput improvement at 1024 labels. Entity embeddings can be pre-computed.
3. **Poly-encoder (2024):** Bi-encoder + post-fusion step for inter-label interactions. Better disambiguation of similar entity types.
4. **Modern-GLiNER (2025):** Uses ModernBERT-large as text encoder, supporting up to 8,192 tokens and 4x better efficiency than DeBERTa-based models.

### Published Papers on GLiNER Fine-Tuning
- **GLiNER-BioMed** (arxiv 2504.00676): Domain-adapted GLiNER for biomedical NER using synthetic data distilled from LLMs + fine-tuning. Achieved 5.96% F1 improvement over strongest baseline in zero-shot and few-shot biomedical NER.
- **GLiNER multitask** (arxiv 2406.12925): Extended GLiNER to handle NER, relation extraction, summarization, QA. Includes experiments on self-learning for NER.
- **"The Million-Label NER"** (arxiv 2602.18487, 2026): Breaking scale barriers with GLiNER bi-encoder architecture.
- **No published paper specifically on LoRA fine-tuning of GLiNER was found on arxiv.**

### LoRA Support in GLiNER
- **GLiNER2** (from fastino-ai/GLiNER2) has built-in LoRA support with configurable parameters: `use_lora=True`, `lora_r` (rank: 4, 8, 16, 32), `lora_alpha`, `lora_dropout`, and `save_adapter_only=True`
- Original GLiNER library supports fine-tuning via `train.py` but LoRA is not a first-class documented feature
- A user-published model `CHFLTM/gliner2-lora-custom` exists on HuggingFace (GLiNER2 + LoRA custom adapter)
- The `gliner-finetune` package (by wjbmattingly) provides synthetic data generation + fine-tuning pipeline

---

## 2. GLiNER + LoRA + Active Learning: Has This Been Published?

**Finding: NO. This specific triple combination has NOT been published.**

After extensive searching across arxiv, Google Scholar results, and HuggingFace:
- No paper combines all three: GLiNER + LoRA + active learning
- GLiNER multitask paper (2406.12925) mentions "self-learning approaches for NER" but this is self-training (pseudo-labeling), NOT active learning with human-in-the-loop annotation
- Active learning for NER exists as a research area (e.g., "Deep Active Learning for Named Entity Recognition," arxiv 1707.05928) but predates GLiNER and LoRA
- LoRA for NER exists (e.g., "Instruction Finetuning LLaMA-3-8B Using LoRA for Financial NER," arxiv 2601.10043) but doesn't involve active learning
- No paper combines PEFT/LoRA with active learning for any NER model specifically

**This represents a genuine research gap and potential novelty.**

---

## 3. PEFT Methods for NER: Standard Approaches

### Key Finding: Full Fine-Tuning Often Beats PEFT for NER
A study on multilingual encoder models (arxiv 2501.06025) found:
- **Full fine-tuning outperforms PEFT methods for NER** across most languages
- Pfeiffer adapters (bottleneck adapters) matched full fine-tuning only for German NER
- **LoRA showed the weakest performance** for NER specifically
- Reasoning: "For word-level tasks, a larger learning capacity is more crucial than preserving fine-grained capabilities from pre-training"
- Contrast: PEFT methods work better for extractive QA tasks

### PEFT Method Comparison (General)
| Method | Trainable Params | Memory (7B model) | Quality vs Full FT |
|--------|-----------------|--------------------|--------------------|
| LoRA | ~0.1-1% | 16-17 GB | 90-95% |
| QLoRA | ~0.1-1% | 6-10 GB | 88-93% |
| Adapters | ~1-3% | 16.6 GB | 92-97% |
| Prefix Tuning | ~0.1% | 14-16 GB | 85-90% |
| Prompt Tuning | ~0.01% | 14 GB | 80-90% |

### NER-Specific LoRA Usage
- Recent NER work with LLMs (Qwen, Llama series) uses LoRA through LLaMA-Factory
- Rank r=4-16 recommended for NLP tasks
- DoRA (2024, NVIDIA) outperforms standard LoRA across tasks
- **For smaller encoder models like GLiNER, higher LoRA rank may be needed**

### Implication for This Project
The finding that LoRA underperforms full fine-tuning for NER is important context. However:
- GLiNER is NOT a standard token classification model; it uses a different span-based architecture
- Active learning reduces data requirements, which may change the PEFT vs full FT calculus
- The practical benefit (small adapter size, quick iteration) may outweigh marginal F1 loss

---

## 4. GLiNER2 vs GLiNER-Multitask

### GLiNER-Multitask (arxiv 2406.12925)
- **Tasks:** Open NER, Relation Extraction, Summarization, QA, Open IE
- **Architecture:** Token-based (not span-based) processing for long-form generation tasks
- **NER Performance (CrossNER + MIT benchmarks):**
  - CrossNER_AI: 51.05%
  - CrossNER_literature: 68.96%
  - CrossNER_music: 74.30%
  - CrossNER_politics: 78.27%
  - CrossNER_science: 67.29%
  - MIT-movie: 56.60%
  - MIT-restaurant: 43.51%
  - **Average: 62.76%**
- Best NER-only average among tested models at the time
- Self-learning improved CrossNER_AI from 51.05% to 63.25% (12.2 point gain)

### GLiNER2 (arxiv 2507.18546, EMNLP 2025 Demos)
- **Tasks:** NER, Text Classification, Hierarchical Structured Data Extraction
- **Key feature:** Multi-task composition in a SINGLE forward pass
- **Architecture:** Pretrained transformer encoder with schema-driven interface
- **Built-in LoRA support** for parameter-efficient fine-tuning
- **Performance:** Competitive with GPT-4o (0.590 vs 0.599 overall F1), better in some domains (AI: 0.547 vs 0.526)
- Open source at fastino-ai/GLiNER2

### Key Differences
| Feature | GLiNER-Multitask | GLiNER2 |
|---------|-----------------|---------|
| Tasks | NER, RE, Summarization, QA, IE | NER, Classification, Structured Extraction |
| Multi-task in one pass | No | Yes |
| LoRA support | Not documented | Built-in |
| Summarization | Yes | No |
| Hierarchical extraction | No | Yes |
| Publisher | Knowledgator | Fastino Labs |

---

## 5. MIT Movies Dataset: State-of-the-Art F1 Scores

### Known Benchmark Results

| Model | MIT Movie F1 | Type |
|-------|-------------|------|
| **UniversalNER-7B** | **90.17%** | Supervised fine-tuning (7B LLM) |
| InstructUIE-11B | 89.58% | Supervised fine-tuning (11B) |
| BERT-base | 88.78% | Supervised fine-tuning |
| GLiNER-multitask-v0.5 | 56.60% | Zero-shot |
| GLiNER-community-large-v2.5 | 53.00% | Zero-shot |
| GLiNER-bi-large-v2.0 | 51.00% | Zero-shot |
| modern-gliner-bi-large-v1.0 | 47.60% | Zero-shot |

### Analysis: Is 86-88 F1 Competitive?
- **86-88 F1 would be competitive with BERT-base supervised (88.78%)** and close to InstructUIE-11B (89.58%)
- It would be **below UniversalNER-7B (90.17%)** but that model is 7 billion parameters
- For a small encoder model (~300M params) with LoRA fine-tuning and active learning, **86-88 F1 would be a strong result**
- Zero-shot GLiNER achieves only ~47-57% on MIT Movies, so reaching 86-88 with fine-tuning represents a massive improvement
- **The gap from zero-shot (~50%) to supervised BERT (~89%) is where this project's contribution lies**

### Dataset Characteristics
- 9,500 training sentences, 22k spans
- 2,500 test sentences, 5.67k spans
- 12 entity classes: ACTOR, YEAR, TITLE, GENRE, DIRECTOR, SONG, PLOT, REVIEW, CHARACTER, RATING, RATINGS_AVERAGE, TRAILER
- Known to have annotation quality issues (~4% of spans are problematic per Galileo AI research)

---

## 6. knowledgator/modern-gliner-bi-large-v1.0

### Confirmed to Exist on HuggingFace
- **Text Encoder:** ModernBERT-large
- **Entity Label Encoder:** BGE-base-en-v1.5
- **Context Length:** Up to 8,192 tokens
- **Efficiency:** 4x better than DeBERTa-based models

### Reported Performance
- **Standard NER Average:** 49.7% (across 19 datasets)
- **Zero-shot CrossNER + MIT Average:** 59.8%
- Notable scores: WikiNeural 83.7%, CoNLL 2003 69.3%, MIT-movie 47.6%
- **MIT-movie zero-shot: 47.6%** (comparable to other GLiNER variants)

### Training Data
1. numind/NuNER (2M samples)
2. knowledgator/GLINER-multi-task-synthetic-data
3. urchade/pile-mistral-v0.1

---

## Summary: Research Gap Analysis

### What Exists
- GLiNER (multiple architectures, actively developed)
- LoRA fine-tuning of GLiNER2 (built-in support)
- Active learning for NER (older literature, pre-GLiNER)
- LoRA for NER with LLMs (recent papers)
- Self-learning/pseudo-labeling with GLiNER (multitask paper)

### What Does NOT Exist (Novel Contributions)
1. **GLiNER + LoRA + Active Learning combined** -- NO published paper found
2. **Active learning with any PEFT method for NER** -- extremely limited literature
3. **Systematic study of LoRA effectiveness specifically for GLiNER architecture** -- no paper found
4. **Active learning to bridge the zero-shot to supervised gap for GLiNER** -- novel framing

### Positioning Recommendation
The project occupies a genuine research gap. The novelty lies in:
- Combining parameter-efficient fine-tuning (LoRA) with active learning for a generalist NER model
- Demonstrating that strategic sample selection can compensate for LoRA's known NER performance gap vs full fine-tuning
- Practical contribution: enabling domain adaptation of GLiNER with minimal labeled data and compute
