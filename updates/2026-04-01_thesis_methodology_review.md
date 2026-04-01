# Thesis Methodology Review: NER with GLiNER + Knowledge Distillation

**Date:** 2026-04-01, ~afternoon

---

## 1. Single Dataset Evaluation (MIT Movies Only)

### Verdict: SIGNIFICANT WEAKNESS

Evaluating on only one dataset is broadly considered insufficient for a research paper in NLP. Here is the evidence:

- **ACL Rolling Review (ARR) Author Guidelines** explicitly state that authors should "state clearly what was done (e.g., what kinds of models/datasets/languages etc were tested)" and identify "where (within the target scope) there is a reasonable expectation that the reported findings may not hold." A single dataset makes it impossible to discuss generalization.
- **Standard NER practice** is to evaluate on at least 2-3 benchmark datasets. The canonical choices are CoNLL-2003, OntoNotes 5.0, and WNUT-2017. The original GLiNER paper (NAACL 2024) evaluated on **20 NER datasets**.
- **Generalization is a core theme** for ACL 2025, described as "critically important for models used in real-world applications." A single-dataset evaluation directly contradicts this.
- Research has shown that models trained on one dataset (e.g., SQuAD) do not transfer to other similar datasets (CoQA, QuAC), even when all are based on Wikipedia. This principle applies equally to NER.

**Recommendation for the thesis:** At minimum, add CoNLL-2003 and one domain-specific dataset (e.g., WNUT for emerging entities, or a biomedical NER dataset). This would transform a significant weakness into an adequate evaluation.

---

## 2. MIT Movies Dataset Assessment

### Verdict: LEGITIMATE BUT NICHE; NOT A FLAGSHIP BENCHMARK

- **Origin:** MIT Movies is a spoken language understanding (SLU) / slot-filling dataset from MIT CSAIL. It was designed for dialogue systems, not as a primary NER benchmark.
- **Size:** ~9,500 training sentences (22k spans), ~2,500 test sentences (5.67k spans), with 12 entity classes (ACTOR, YEAR, TITLE, GENRE, DIRECTOR, SONG, PLOT, REVIEW, CHARACTER, RATING, RATINGS_AVERAGE, TRAILER).
- **Known quality issues:** Galileo AI's analysis found ~3% of spans are problematic (incorrect boundaries, wrong tags, untagged entities). Fixing just 4% of span data improved F1 by up to 3.3 points, suggesting noisy annotations.
- **Class overlap:** Several classes are semantically overlapping, which creates ambiguity for both annotators and models.
- **Standing in the community:** It is used in some NER/SLU papers but is NOT one of the standard NER benchmarks (those are CoNLL-2003, OntoNotes 5.0, WNUT, Few-NERD). It is closer to a "domain-specific" or "niche" dataset than a toy dataset, but its annotation quality issues weaken conclusions drawn from it.

**Conclusion:** Using MIT Movies is not inherently wrong, but relying on it *exclusively* is problematic. It is not considered a gold-standard benchmark, and its annotation noise means F1 differences of 1-3 points may be within the noise margin.

---

## 3. Gemma 3 12B as a Teacher Model

### Verdict: QUESTIONABLE CHOICE; WEAKER THAN ALTERNATIVES

- **General capability:** Gemma 3 12B-IT scores 9 on the Artificial Analysis Intelligence Index, placing it at the lower end among comparable models. The 27B variant closely approaches Gemini 1.5 Pro, but the 12B is significantly weaker.
- **NER-specific issues with LLMs:** Research (GPT-NER, 2023) established that "performance on NER is still significantly below supervised baselines" for generative LLMs because NER is a sequence labeling task while LLMs are text-generation models. This gap is worse for smaller models like Gemma 3 12B.
- **Gemma NER benchmarks:** Gemma models show high variability across NER datasets. In medical NER, Gemma achieved AVG_MICRO of 0.9962 on one dataset but only 0.8029 on another. In document extraction, Gemma 3 27B achieved only F1 of 41.3 in zero-shot settings.
- **Compared to GPT-4:** GPT-4 consistently outperforms open-source models on zero-shot NER. GPT-4 achieved F1 of 71.3 on BC5 and 58.4 on NCBI in zero-shot. GPT-3.5 achieved F1 of 73.4 on CoNLL-2003 zero-shot. Gemma models typically score 10-30 points lower.
- **Terms of service consideration:** One legitimate reason to avoid GPT-4 as teacher is that OpenAI's ToS historically barred using outputs to train competing models. However, this is less of an issue for academic research, and Gemma's own distillation was done from a larger Gemini model.

**Better alternatives would have been:**
1. GPT-4 or GPT-4o (highest quality annotations, if ToS permits for academic use)
2. Gemma 3 27B (same family, but significantly stronger)
3. A specialized NER model like UniNER-7B as a teacher (purpose-built for the task)
4. An ensemble approach using multiple teacher models

**Key concern:** If the teacher model only achieves ~70 F1, the ceiling for the student model is inherently limited by annotation quality. Garbage in, garbage out.

---

## 4. Optuna + 2500 Training Steps

### Verdict: BORDERLINE ACCEPTABLE; NEEDS BETTER JUSTIFICATION

**The 2500 steps:**
- The codebase shows `num_steps: 2500` with `eval_steps: 100` and `patience: 7`, meaning early stopping after 700 steps without improvement. Effective training may be much less than 2500 steps.
- For a dataset of ~9,500 sentences with batch size 8, one epoch is ~1,187 steps. So 2500 steps is roughly 2 epochs. This is reasonable for LoRA fine-tuning, where overfitting on small datasets is a real risk.
- Research suggests that for small instruction datasets, single-pass fine-tuning with stronger regularization is preferred to avoid overfitting.
- The learning rate (0.00022) is within the typical LoRA range (5e-6 to 2e-4), though on the higher end.

**The Optuna search:**
- Using Optuna for hyperparameter search is a legitimate and well-established practice. Research shows "there is no single rule of thumb for tuning LoRA hyperparameters" based on 1000+ experiments.
- The hyperparameters in the config (e.g., `learning_rate: 0.00022105770821309302`, `warmup_ratio: 0.15560507652730393`) show the telltale precision of Optuna optimization, which is good practice.
- **However:** The thesis should report the number of Optuna trials, the search space, and the optimization objective. Without this, it is not reproducible.

**LoRA config assessment:**
- `r=64, lora_alpha=128` (alpha = 2x rank) follows the recommended heuristic for aggressive learning. This is on the higher end -- lower ranks (8-16) are typical for fine-tuning on a base model, while 32-64 is for teaching new concepts. For NER domain adaptation, r=16-32 might have been sufficient.
- The target modules list is extensive, covering attention layers, FFN layers, and GLiNER-specific span representation layers. This is thorough.

---

## 5. F1 Score Reasonableness (GLiNER ~88, Gemma ~70)

### Verdict: PLAUSIBLE BUT NEEDS CONTEXT

**GLiNER ~88 F1 with LoRA fine-tuning:**
- The original GLiNER paper reports zero-shot F1 of ~55-60 across diverse datasets. Fine-tuned GLiNER on domain-specific data has achieved 80+ F1 in cybersecurity NER (80.5 F1).
- An 88 F1 on MIT Movies with supervised LoRA fine-tuning is plausible, especially given that MIT Movies has relatively constrained entity types and the model is being fine-tuned on in-domain data.
- However, the annotation noise in MIT Movies (3% problematic spans) means the true ceiling may be lower than reported. An 88 F1 might partly reflect learning the annotation noise.

**Gemma 3 12B ~70 F1 zero-shot:**
- GPT-3.5 achieves ~73 F1 on CoNLL-2003 zero-shot. GPT-4 achieves ~71 F1 on medical NER zero-shot.
- Gemma 3 12B is generally weaker than GPT-4 on structured tasks. A 70 F1 on a relatively simple domain (movies) seems on the high side for Gemma 3 12B -- this should be verified carefully.
- If the 70 F1 is accurate, it suggests Gemma 3 12B is being used near its maximum capability for this task, which raises questions about annotation quality when used as a teacher.

**The ~18 point gap (88 vs 70):**
- An 18 F1-point improvement from zero-shot LLM to fine-tuned specialist is a reasonable magnitude. This is the expected benefit of domain-specific fine-tuning.
- However, the comparison is somewhat unfair: zero-shot LLM vs. supervised fine-tuned model. A more rigorous comparison would include few-shot LLM, fine-tuned LLM, and other supervised NER baselines (e.g., fine-tuned BERT/RoBERTa).

---

## 6. Cost-Benefit Analysis as Contribution

### Verdict: INCREASINGLY COMMON AND VALUABLE; NOT NOVEL BY ITSELF

- **Precedent exists:** The paper "Scaling Down to Scale Up: A Cost-Benefit Analysis of Replacing OpenAI's LLM with Open Source SLMs in Production" (2023) directly addresses LLM API costs vs. fine-tuned smaller models, and has been well-received.
- **Break-even analysis** is an established framework: research shows there is a "usage threshold at which ChatGPT is more cost-effective than utilizing open-source LLMs deployed to AWS."
- **Industry relevance:** Cost analysis is particularly valued in industry-track papers (EMNLP 2025 industry track explicitly welcomes such contributions).
- **As a standalone contribution:** A cost analysis alone is insufficient for a research paper. It should accompany a technical contribution (novel method, architecture, training approach).
- **For a thesis:** This is a perfectly appropriate secondary contribution. It demonstrates practical thinking and real-world applicability.

**What would strengthen it:** Including actual dollar figures, GPU hours, carbon footprint, and a clear break-even analysis at different annotation volumes.

---

## 7. Methodological Red Flags for NLP Reviewers (2024-2025)

Based on ARR reviewer guidelines (updated October 2024) and recent meta-research on peer review:

### Red Flags Applicable to This Thesis:

1. **Single dataset evaluation** -- The most critical issue. Reviewers routinely flag this as "Extra Experiments" needed, and it is the most frequent criticism category in NLP reviews.

2. **Claims not supported by evidence** -- ARR guidelines warn about "claims that are not actually supported by the evidence or by the arguments, but that are presented as conclusions rather than as hypotheses/discussion." If the thesis claims generalizability from MIT Movies alone, this is a red flag.

3. **Missing baselines** -- Not comparing against standard NER baselines (fine-tuned BERT, RoBERTa, SpanBERT) is a significant omission.

4. **Reproducibility gaps** -- ARR specifically assesses "whether enough details/code is provided for reproducibility." Optuna search spaces, random seeds, and exact training configurations must be documented.

5. **Responsible NLP checklist violations** -- Since December 2024, egregious checklist violations lead to desk rejections. Compute budget, environmental impact, and dataset licenses should be documented.

### General Red Flags in 2024-2025 NLP Papers:

- **Hallucinated references** -- A major concern with LLM-assisted writing
- **Anonymity violations** -- Links to non-anonymous repos, self-citations
- **Insufficient error analysis** -- Just reporting F1 without analyzing failure modes
- **Cherry-picked examples** -- Showing only successes without representative failures
- **Overclaiming from limited evidence** -- Making broad claims from narrow experiments
- **Not engaging with concurrent/recent work** -- Failing to cite or compare with recent relevant papers

---

## Summary of Critical Issues (Ranked by Severity)

| # | Issue | Severity | Fixable? |
|---|-------|----------|----------|
| 1 | Single dataset evaluation | HIGH | Yes -- add 2-3 more datasets |
| 2 | Weak teacher model (Gemma 3 12B) | MEDIUM-HIGH | Partially -- at least acknowledge and compare |
| 3 | Missing standard baselines | MEDIUM-HIGH | Yes -- add BERT/RoBERTa NER baselines |
| 4 | MIT Movies not a flagship benchmark | MEDIUM | Partially -- add standard benchmarks alongside |
| 5 | Insufficient Optuna documentation | MEDIUM | Yes -- document search space and trials |
| 6 | No error analysis | MEDIUM | Yes -- add confusion matrix, failure examples |
| 7 | Unfair comparison (zero-shot vs fine-tuned) | LOW-MEDIUM | Yes -- add few-shot LLM baseline |

---

## Sources

- [ARR Reviewer Guidelines](http://aclrollingreview.org/reviewerguidelines)
- [ARR Authors Guidelines](http://aclrollingreview.org/authors)
- [ARR Common Submission Problems](http://aclrollingreview.org/authorchecklist)
- [ARR Responsible NLP Checklist](http://aclrollingreview.org/responsibleNLPresearch/)
- [EMNLP 2025 Industry Track CFP](https://2025.emnlp.org/calls/industry_track/)
- [GLiNER: NAACL 2024 Paper](https://aclanthology.org/2024.naacl-long.300/)
- [GLiNER GitHub](https://github.com/urchade/GLiNER)
- [MIT Movies Dataset - Kaggle](https://www.kaggle.com/datasets/dmitrytronin/mit-movies-ner)
- [MIT Movies Fixed Format - HuggingFace](https://huggingface.co/datasets/rungalileo/mit_movies_fixed_connll_format)
- [Galileo: Improving NER Datasets](https://galileo.ai/blog/improving-your-ml-datasets-part-2-ner)
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
- [Gemma 3 12B - Artificial Analysis](https://artificialanalysis.ai/models/gemma-3-12b)
- [GPT-NER: NER via LLMs](https://arxiv.org/abs/2304.10428)
- [Scaling Down to Scale Up: Cost-Benefit Analysis](https://arxiv.org/html/2312.14972v3)
- [Understanding LLM Fine-Tuning Costs](https://arxiv.org/abs/2408.04693)
- [Optuna for LLM Fine-Tuning](https://www.newline.co/@zaoyang/how-to-use-optuna-for-llm-fine-tuning--f702db18)
- [LoRA Hyperparameters Guide - Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Practical Tips for LoRA - Sebastian Raschka](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [12 Best Practices for Distilling from GPT](https://predibase.com/blog/graduate-from-openai-to-open-source-12-best-practices-for-distilling-smaller)
- [LazyReview: Uncovering Lazy Thinking in NLP Reviews](https://arxiv.org/html/2504.11042v1)
- [Familiarity: Zero-Shot NER Evaluation](https://arxiv.org/html/2412.10121v2)
- [NER Datasets Classification Framework](https://link.springer.com/article/10.1007/s44196-024-00456-1)
- [NLP-progress: NER](http://nlpprogress.com/english/named_entity_recognition.html)
- [EMNLP 2024 Ethical Review](https://2024.emnlp.org/Strengthening-Ethical-Review-in-Natural-Language-Processing-Insights,-Best-Practices,-Resources-and-Paths-Forward/)
