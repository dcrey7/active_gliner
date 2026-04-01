# Literature Review: Synthetic Data & Synthetic Labels for NER

**Date:** 2026-04-01, ~14:00 UTC

---

## 1. LLM-Generated Labels to Train Smaller NER Models (Teacher-Student / Distillation Pipeline)

### Has the pipeline "LLM generates labels -> train small model with LoRA" been published?

**Yes, this general pipeline is well-established, though the exact combination varies across papers.**

#### UniversalNER (Zhou et al., ICLR 2024) - arxiv:2308.03279
- **Pipeline:** ChatGPT (gpt-3.5-turbo) annotates 50,000 passages from the Pile corpus -> generates 45,889 input-output pairs with 240,725 entities across 13,020 entity types -> fine-tunes LLaMA 7B/13B student models via instruction tuning
- **Key results:** UniNER-7B achieves 41.7% zero-shot F1 (vs ChatGPT's 34.9%), outperforming the teacher by 7-9 F1 points. Outperforms Alpaca/Vicuna by 30+ F1 points
- **LoRA:** NOT explicitly used; appears to use full instruction tuning
- **Benchmark:** 43 NER datasets across 9 domains (biomedicine, programming, social media, law, finance)
- **Status:** VERIFIED. Published at ICLR 2024. This is the seminal paper for NER distillation from LLMs

#### GLiNER-biomed (Yazdani et al., April 2025) - arxiv:2504.00676
- **Pipeline:** OpenBioLLM-70B annotates 10,000 samples -> **LoRA fine-tunes** an 8B distilled model -> 8B model annotates 105,000 additional passages (2.3M entity mentions, 640K unique entities) -> trains GLiNER models (DeBERTa-v3 small/base/large)
- **Key results:** 59.77% F1 zero-shot (vs 53.81% previous SOTA), +5.96 F1 points improvement
- **LoRA:** YES - explicitly used for distilling the 70B model into the 8B model
- **This is the closest published paper to the "LLM labels -> LoRA fine-tune smaller model for NER" pipeline**

#### KD-LoRA (Azimi et al., NeurIPS ENLSP-IV 2024) - arxiv:2410.20777
- **Pipeline:** Full fine-tune teacher model -> inject LoRA into smaller student model -> knowledge distillation transfer
- **Key results:** 75% less GPU memory vs FFT, 30% less than LoRA alone, ~40% more compact model, ~30% faster inference
- **Evaluated on:** BERT, RoBERTa, DeBERTaV3 (encoder-only models)
- **Not NER-specific** but directly applicable architecture

#### Rapid Adaptation of Chemical NER (2025) - J. Chem. Inf. Model.
- **Pipeline:** LLM-annotated data -> few-shot metric-learning NER model
- **Key results:** Reasonable performance with only 5 examples per entity type; distills LLM knowledge into lightweight model for efficient in-house use
- **Domain:** Chemistry/materials science

#### LLM-NER with LoRA+ (2024) - IEEE
- **Pipeline:** Fine-tunes Meta-Llama-3-8B-Instruct with LoRA+ for NER
- **Key results:** Surpasses benchmarks by 10% in F1 score

#### Financial NER with LoRA (Jan 2026) - arxiv:2601.10043
- **Pipeline:** LLaMA-3-8B + instruction fine-tuning + LoRA for financial NER
- **Key results:** Micro-F1 of 0.894 on 1,693 sentences

### Summary for Q1
The "LLM generates labels -> train smaller model" pipeline is well-established (UniversalNER is the landmark paper). The specific addition of LoRA for efficient fine-tuning in this pipeline has been published (GLiNER-biomed, KD-LoRA). However, the specific combination of "LLM generates NER labels -> LoRA fine-tune a GLiNER-style model" appears to be a narrower niche that is only now being explored.

---

## 2. Verification of Specific Papers

### ProgGen (Heng et al., 2024) - arxiv:2403.11103
- **Status:** VERIFIED. Published at ACL 2024 Findings
- **Authors:** Yuzhao Heng, Chunyuan Deng, Yitong Li, Yue Yu, Yinghao Li, Rongzhi Zhang, Chao Zhang
- **Approach:** LLMs self-reflect on domain -> generate domain-relevant attributes -> generate entity terms first -> then create NER context around entities (bypasses LLM difficulty with structured output)
- **Datasets:** CoNLL-2003, WikiGold, MIT-Movie, MIT-Restaurant
- **LLM used:** OpenAI API (likely GPT-3.5/GPT-4)
- **Downstream models:** BERT-based models trained on synthetic data
- **Key finding:** Significant improvements over conventional data generation methods; more cost-effective
- **Mixing study:** No explicit mixing ratio ablation reported
- **Code:** https://github.com/StefanHeng/ProgGen

### GPT3Mix (Yoo et al., 2021) - arxiv:2104.08826
- **Status:** VERIFIED. Published at EMNLP 2021 Findings
- **Authors:** Kang Min Yoo, Dongju Park, Jaewook Kang, Sang-Woo Lee, Woomyoung Park (NAVER AI Lab)
- **Approach:** Uses GPT-3 to generate augmented text samples from a mixture of real samples; uses soft-labels predicted by GPT-3 for knowledge distillation
- **Tasks:** Text CLASSIFICATION only (SST-2, TREC, CR, etc.) -- NOT NER/sequence labeling
- **Key innovation:** Soft-label distillation from LLM label-token distributions
- **IMPORTANT NOTE:** This paper does NOT cover NER. It is relevant as a conceptual predecessor for using LLM soft-labels, but it operates on classification tasks only

### GuideX (De La Fuente et al., 2025) - arxiv:2506.00649
- **Status:** VERIFIED. Published at ACL Findings 2025
- **Authors:** Neil De La Fuente, Oscar Sainz, Iker Garcia-Ferrero, Eneko Agirre
- **Approach:** Automatically defines domain-specific schemas, infers guidelines, generates synthetically labeled instances for zero-shot IE
- **Student model:** Llama 3.1 fine-tuned with GuideX synthetic data
- **Key results:**
  - Without human data: +7 F1 over previous methods across 7 zero-shot NER benchmarks
  - With human data combined: +2 F1 over previous methods
- **Code:** https://neilus03.github.io/guidex.com

### UniversalNER (Zhou et al., 2024) - arxiv:2308.03279
- **Status:** VERIFIED. Published at ICLR 2024
- **Details:** See Section 1 above for full analysis
- **Key claim verified:** Distilled student outperforms ChatGPT teacher by 7-9 F1 points

---

## 3. Is the "75% Ground Truth + 25% Synthetic = 100% Ground Truth" Finding Known?

### Short answer: This specific ratio has NOT been published as a well-known result, but related findings exist.

#### What the literature shows about mixing ratios:

1. **"Does Synthetic Data Help NER for Low-Resource Languages?"** (arxiv:2505.16814, May 2025)
   - Finding: "Even 100 manually annotated datapoints can yield NER models that cannot be matched by models trained on much larger amounts of synthetic data"
   - GPT-4 produced 97.0% usable datapoints vs 59.3% for Llama-3.1
   - **No specific mixing ratio equivalence found** -- organic data consistently outperformed synthetic

2. **Golden Ratio Mixing** (He et al., 2025)
   - Theoretical result: optimal real-data weight ~0.618 (golden ratio reciprocal) for balanced samples
   - Derived from recursive generative modeling analysis to prevent model collapse

3. **Object Detection / Tracking** (Chang et al., 2024)
   - "Substituting 60-80% of real data with synthetic data incurs negligible loss" when synthetic generator is well-tuned
   - In object detection: mixed datasets with 5-20% real data often match or beat pure real data

4. **ASR** (DeRenzi et al., 2025)
   - ASR achieved parity with 100% real data at 1:1 or 1:2 real-to-synthetic ratio

5. **LLM Pre-training** (2025)
   - Both 33% and 67% synthetic mixtures with CommonCrawl yield similar performance for rephrased data
   - Textbook-style mixtures favor less synthetic data (33% >> 67%)

6. **General finding** (Shidani et al., 2025)
   - U-shaped risk curve with respect to synthetic data proportion
   - Moderate inclusion is favored, especially if generator deviates from target distribution

7. **JMIR 2025 - Estonian Medical NER** (Suvalov et al.)
   - Pipeline: GPT-2 generates synthetic health data -> GPT-3.5/GPT-4 annotate -> fine-tune NER model
   - Best F1: 0.69 for drug extraction, 0.38 for procedure extraction
   - Explored relationship between amount of synthetic data and performance

### Conclusion for Q3
The specific "75% GT + 25% synthetic matches 100% GT" finding would be a **novel contribution** if demonstrated rigorously. The literature shows the optimal ratio is task/domain-dependent, and most studies find that real data is disproportionately more valuable than synthetic data for NER specifically. The closest known result is the "golden ratio" principle (~62% real, ~38% synthetic) but this is theoretical and not NER-specific.

---

## 4. Current Best Practices: LLMs as Annotators

### Key Survey
**"Large Language Models for Data Annotation and Synthesis: A Survey"** (Tan et al., EMNLP 2024 Oral)
- arxiv:2402.13446
- Comprehensive taxonomy of LLM annotation: generation, assessment, utilization
- GitHub: https://github.com/Zhen-Tan-dmml/LLM4Annotation

### Evidence that LLM Annotations Are Sufficient

1. **GPT-4 label quality (Refuel AI technical report):**
   - 88.4% agreement with ground truth across multiple NLP tasks
   - Human annotators: 86.2% agreement
   - GPT-4 out-of-the-box EXCEEDS typical crowdsourced human annotator quality

2. **Clinical NER with prompt engineering (JAMIA 2024):**
   - GPT-4 with enhanced prompts: F1 = 0.861 (MTSamples), 0.736 (VAERS)
   - Still below BioClinicalBERT fine-tuned (F1 = 0.901, 0.802)
   - Gap is closing with better prompting

3. **AnnoLLM (He et al., NAACL 2024):**
   - Explain-then-annotate prompting strategy
   - "Surpasses or performs on par with crowdsourced annotators" across tested tasks

4. **LLMs outperform outsourced human coders** (Nature Scientific Reports 2025):
   - Across tasks including NER, LLMs consistently outperform outsourced human coders
   - Especially strong in tasks requiring deep contextual understanding

5. **FiNERweb (arxiv:2512.13884, Dec 2025):**
   - GPT-4o-mini + Gemma3-27B annotations merged for multilingual NER
   - 225K passages, 235K distinct entity labels across 91 languages
   - Models trained on FiNERweb match or improve zero-shot transfer despite 19x less data
   - LLM-as-a-judge confirms high faithfulness and completeness

### Current Best Practices Summary
- Use GPT-4/GPT-4o for highest quality annotations; Gemma-27B+ for cost-effective alternatives
- Explain-then-annotate or few-shot chain-of-thought prompting improves quality
- Merge annotations from multiple LLMs for higher coverage (FiNERweb approach)
- Use noise-aware training when incorporating LLM annotations
- Small amounts of human-verified data remain valuable as anchors
- LLM annotations are now considered sufficient for training competitive NER models, especially when combined with even small amounts of human validation

---

## 5. Papers Combining ALL THREE: Active Learning + Synthetic Labels + Efficient Fine-Tuning for NER (2024-2025)

### Direct combination of all three: NO single paper found that combines all three for NER.

This appears to be a **gap in the literature** as of April 2026. Here is what exists for each pair:

#### Active Learning + LLM Annotation (closest to the full combination):

1. **"LLMs in the Loop" (Kholodna & Julka, ECML PKDD 2024)** - arxiv:2404.02261
   - LLMs (GPT-4-Turbo) in active learning loop for NER annotation
   - Evaluated on MasakhaNER 2.0 (20 African languages)
   - 42.45x cost savings vs human annotation
   - Near-SOTA performance with reduced data
   - **Missing:** No LoRA/PEFT; no synthetic data generation

2. **"Survey of LLM-based Active Learning"** (ACL 2025 Long) - arxiv:2502.11767
   - Comprehensive survey classifying methods by querying and annotation processes
   - LLMs enable both query generation and annotation
   - Covers text classification primarily, some NER references

3. **Active Learning for Clinical NER** (PMC, Oct 2024)
   - Dynamic AL strategies including diversity-based (CLUSTER) and hybrid approaches
   - **Missing:** No LLM-based labeling; no PEFT

#### Active Learning + Synthetic Data Generation:

4. **"Towards Active Synthetic Data Generation for Finetuning Language Models"** (arxiv:2512.00884, Dec 2025)
   - Iterative, closed-loop synthetic data generation guided by student model state
   - Active learning selection criteria used for data generation
   - **Missing:** Evaluated on math/reasoning tasks, not NER; LoRA not mentioned

#### Synthetic Labels + Efficient Fine-Tuning for NER:

5. **GLiNER-biomed** (see Section 1) - Uses LoRA for distillation + synthetic NER data
6. **UniversalNER** - Distillation + synthetic labels, but full fine-tuning
7. **GuideX** - Synthetic data generation + fine-tuning Llama 3.1, but not explicitly PEFT

#### Active Learning + Efficient Fine-Tuning:

8. **Active Learning with PLMs for NER in Requirements Engineering** (ScienceDirect, 2024)
   - Reduces labeling effort by 74% while improving performance
   - **Missing:** No synthetic data; no LoRA specifically

### The Gap
**No published paper (as of early 2026) combines all three of: (1) active learning for sample selection, (2) LLM-generated synthetic labels/data, and (3) parameter-efficient fine-tuning (LoRA) specifically for NER.** The closest are:
- "LLMs in the Loop" (active learning + LLM annotation for NER, but no PEFT)
- GLiNER-biomed (synthetic labels + LoRA for NER, but no active learning)
- "Active Synthetic Data Generation" (active learning + synthetic data + fine-tuning, but not NER)

**This represents a clear novelty opportunity.**

---

## Summary Table

| Paper | LLM Labels | Train Small Model | LoRA/PEFT | Active Learning | NER | Mixing Study |
|-------|-----------|-------------------|-----------|-----------------|-----|-------------|
| UniversalNER (ICLR 2024) | Yes (ChatGPT) | Yes (LLaMA 7B/13B) | No | No | Yes | No |
| ProgGen (ACL 2024) | Yes (OpenAI) | Yes (BERT) | No | No | Yes | No |
| GPT3Mix (EMNLP 2021) | Yes (GPT-3 soft) | Yes (classifiers) | No | No | No (classification) | No |
| GuideX (ACL 2025) | Yes (auto-schema) | Yes (Llama 3.1) | Not specified | No | Yes | Partial |
| GLiNER-biomed (2025) | Yes (70B->8B) | Yes (DeBERTa) | Yes (distillation) | No | Yes | No |
| KD-LoRA (NeurIPS-W 2024) | Implicit | Yes | Yes | No | No (general) | No |
| LLMs in the Loop (ECML 2024) | Yes (GPT-4-Turbo) | Yes (classifiers) | No | Yes | Yes | No |
| Active Synthetic Data Gen (2025) | Yes | Yes | Not specified | Yes | No (math) | No |
| FiNERweb (2025) | Yes (GPT-4o+Gemma) | Yes (multiple) | No | No | Yes | No |
| Synthetic NER Low-Resource (2025) | Yes (GPT-4) | Yes | No | No | Yes | Partial |

---

## Key Takeaways for the active_gliner Project

1. **The pipeline is validated but not saturated**: LLM-label -> train-small-model for NER is proven (UniversalNER, GLiNER-biomed), but adding active learning to this pipeline is novel
2. **LoRA for NER distillation is emerging**: GLiNER-biomed (2025) is the first to explicitly use LoRA in the NER distillation pipeline
3. **The 75%/25% mixing finding would be novel**: No published NER study has demonstrated this specific threshold
4. **The triple combination (AL + synthetic + PEFT for NER) is an open research gap**: This is a clear contribution opportunity
5. **LLM annotation quality is now considered sufficient**: Multiple 2024-2025 studies confirm GPT-4-class models match or exceed crowdsourced human annotators
