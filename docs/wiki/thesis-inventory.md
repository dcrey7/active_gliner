---
title: What the Active GLiNER thesis already did
date: 2026-09-25 11:40 CEST
author: Claude, from a full read of docs/master thesis final2.txt (1,963 lines) and results2/
type: wiki
status: living document - edit in place
---

# What the Active GLiNER thesis already did

Source: `docs/master thesis final2.txt`, `results2/*/csv`. Setup: MIT Movies (9,774 train, 2,442 test, 12 types), `knowledgator/modern-gliner-bi-large-v1.0`, 2x RTX 3090 in Docker (via mentor Philippe Fraisse), Gemma 3 12B via Ollama, Qwen3 235B via Cerebras.

## 1. Experiments and findings

| # | Experiment (thesis section, results folder) | What it found |
|---|---|---|
| E1 | Zero-shot GLiNER, per-type report and confidence bins 0-25/26-50/51-75/76-100 (4.1) | 46.6 F1, precision 54.9%. 3,932 predictions, none below 51% (1,608 in 51-75, 2,324 in 76-100); the empty low bins reflect the 0.5 prediction threshold. The thesis reads this as overconfidence; the histogram alone does not prove it. Title, average ratings, review, trailer near 0 F1 |
| E2 | Full fine-tune (3.3) | about 87 F1 |
| E3 | Optuna search: lr, others_lr, both weight decays, warmup, batch, LoRA rank, grad accumulation; median pruning; MLflow (3.4, `exp_gliner_best_hyperparameters`) | best about 88.5 F1 with about 4% trainable parameters; lr and others_lr most sensitive |
| E4 | LoRA layer study, 8 target-module groups (3.4, 4.3, `exp_gliner_best_lora_layers`) | ModernBERT + Task 88.76 best in the CSV (text says All Layers); BERT-only and Task-only far worse |
| E5 | LoRA fine-tune, per-type report and confidence bins (4.1) | about 85.4 F1; 5,953 predictions, 4,923 in 76-100. The thesis reads this as better calibration; histograms alone do not establish it |
| E6 | Threshold sweep 0.1 / 0.5 / 0.9, zero-shot and fine-tuned (4.2, `exp_gliner_llm_threshold_f1`) | fine-tuned: 78.4 / 85.5 / 85.8 F1; zero-shot collapses at 0.9 (22.9). LLM is a single point (69.4) with no threshold |
| E7 | Active learning: min, MNLP, MSE (new), random with results over 14 budgets from 10 to 2,500 (avg implemented, no reported results) (3.5, 4.4, `exp_active_learning_confidence_strategies`) | min reaches 80 F1 at 200 sentences; random 80.2 at 200; MSE and MNLP slower early, catch up after 1,000; all converge above 85 at 2,500 |
| E8 | Hard-example baseline: GLiNER vs Gemma on the worst-N confidence training sentences (4.5, `exp_confidence_gliner_llm_baseline_f1`) | Gemma drops to 56.9 (worst 100) and 60.7 (worst 500) against 69.4 on the full test: shared difficulty, not stated openly in the thesis |
| E9 | Fine-tune on Gemma labels vs ground truth for the same worst-N sentences, 13 budgets (4.5, `exp_confidence_gliner_llm_ft_f1`) | Gemma-label students plateau at 69-70 F1; ground-truth students reach 85.2 at 2,500 |
| E10 | Mixing grid 0/25/50/75/100% ground truth x budgets, min ranking; repeated with MSE ranking (4.6, `exp_confidence_mixed_ft_f1*`) | 0% about 70; 50% high 70s to low 80s; 75% within about 1 F1 of 100% after about 1,000 examples; same shape with MSE |
| E11 | Four teachers (5.1) | Mistral 7B about 64, Gemma 3 12B 69.4, Qwen3 235B instruct about 75, reasoning about 77 F1; errors "qualitatively similar" (not measured) |
| E12 | Fully synthetic text + labels (genre x country control variables, GuideX / ProgGen style), mixed with corrected examples; heatmap (5.2) | synthetic text hurts, most with few corrected examples; synthetic labels on real text are the better route |
| E13 | Prompting: StandardPrompt vs StructuredPrompt (Pydantic schema); validation layer (3.6) | about 100 input and a few hundred to 700 output tokens per sentence; validator checks JSON, schema, substring, types |
| E14 | Cost study: provider table with free and paid tier limits; cost per 1M tasks; GLiNER training on EC2 / SageMaker; 3-year cost (2.4, 4.7) | the thesis states GLiNER about 9.60 EUR over 3 years vs 8,184 EUR for GPT-4o mini; its own 682 EUR per 1M tasks implies 2,046 EUR over 3 years at 1M tasks a year, and 9.60 EUR covers estimated retraining only, not total operating cost |
| E15 | Engineering: LLM backends with rate limits, retries, quotas, cost tracking; on-disk label cache; Label Studio link; modular SOLID code; Docker (3.1, 3.6) | code exists; reuse is checked during the refactor |
| E16 | Discussion: catastrophic forgetting and adapter routing (6.3); agentic routing where low-confidence extractions go to a larger model (8.2); relations, classification, QA as next steps (6.1, 8.1) | broader tasks and agent routing were proposed as future work; the exact four-task design (including slot JSON) is new |

## 2. Problems found while reading (fix in the new paper)

1. Optuna and early stopping used the test set ("evaluation on the full test set used both for early stopping and as the Optuna objective"). The new paper tunes on frozen dev sets only.
2. The pool filter drops sentences with no student prediction or no ground-truth entity (uses hidden labels).
3. LoRA F1 appears as 85.4 / 85.5 / 86.5 / 88.5 / 88.76 across sections; each needs its configuration attached (they may be different configs, not errors). LoRA r16 (text) vs r64 (configs); best layer group All Layers (text) vs ModernBERT + Task (CSV).
4. Section 7.1 says strategies "consistently outperformed random by one to two F1 points"; the E7 table does not show that at small budgets, and there is no reported seed replication.
5. The cost table's LLM F1 column (93-98%) was assumed, not measured.
6. E8 compares training-pool subsets with a full-test score; cumulative worst-N, not disjoint bins.

## 3. How each experiment carries into the new paper

| Thesis | New paper |
|---|---|
| E1, E5 confidence bins | same 4 bins, per task, before and after LoRA (calibration, descriptive) |
| E3, E4 Optuna and layer study | rerun once per task on dev only for GLiNER2.5-multi (section 13), then frozen (code reused) |
| E6 threshold sweep | threshold chosen on dev; sweep reported as descriptive |
| E7 strategies | min vs random (+ diversity) with seeds; MSE dropped (design section 5 cut list), cited as a thesis result |
| E8 hard-example baseline | Q2: disjoint confidence bins, matched sentences, 4 teachers, 6 datasets |
| E9 LLM labels vs ground truth | the core gold-vs-teacher arms |
| E10 mixing grid | adapted replication on MIT Movie: 25% (existing random-mix cell), 50%, 75% ground truth, 6 new runs |
| E11 four teachers | Q3 and Q4 teacher ladder with newest models, errors measured |
| E12 synthetic text | cited as prior finding; not rerun |
| E13 prompts and validator | reused; structured output per task |
| E14 cost study | cost views in section 3f, measured tokens and latency |
| E15 engineering | reused in the tool where the refactor check passes; otherwise rebuilt on gliner2 |
| E16 future work | now the four-task paper (this one) and the agent-decision paper (later) |

## Changelog

- 2026-09-25 11:46 CEST - Codex round 18: seed wording, MSE status, conditional reuse.

- 2026-09-25 11:42 CEST - Corrections from Codex round 17 (E1 precision, E7 counts, E14 arithmetic, calibration and reuse claims).

- 2026-09-25 11:40 CEST - Created from a full read of the thesis text and the results folder.
