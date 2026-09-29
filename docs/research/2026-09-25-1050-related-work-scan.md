---
title: Related work scan for the Active GLiNER paper
date: 2026-09-25 10:50 CEST
author: Claude (research agent), checked against the arXiv API
type: research
status: frozen
---

# Related work scan for the Active GLiNER paper

Scope: active learning for NER, LLM annotators, LLM-to-small-model distillation, NER calibration. arXiv ids were checked against the arXiv API (abstracts only, not full PDFs). Re-check for new work the week before submission.

## Novelty check

+ No arXiv paper selects sentences by GLiNER or GLiNER2 confidence (33 GLiNER hits checked by title and abstract).
+ Closest non-arXiv work: Tafer, Gaio, Lacasta, "A Hybrid Annotation Method and a Reference Corpus for Hiking Descriptions", GeoExT@ECIR 2026, CEUR-WS Vol-4201, paper 11. It uses zero-shot GLiNER pre-annotation, human correction in Argilla, and GLiNER retraining over three rounds. It is a corpus paper, with no selection rule and no random comparison. Cite it.

## Papers

| # | Paper | Id / venue | Finding | Role for us |
|---|---|---|---|---|
| 1 | On the Fragility of Active Learners for Text Classification | 2403.15744, EMNLP 2024 | AL loses to random in over 50% of about 1000 setups | main reviewer threat |
| 2 | When Active Learning Falls Short: Chemical Reaction Extraction | 2604.19335 | uncertainty and diversity rules unstable for NER taggers | threat, closest NER evidence |
| 3 | Do We Still Need Humans in the Loop? | 2604.13899 | full-pool LLM labels beat AL (German hate speech, prefiltered pool) | motivates the full-pool baseline |
| 4 | Scoping Review of AL for Entity Recognition | 2407.03895, DeLTA 2024 | 62 papers; only 13 report run time | supports our cost reporting |
| 5 | Reassessing AL Adoption in NLP | 2503.09701, EACL 2026 | blockers: setup effort, unclear savings, tooling | supports the tool |
| 6 | PEFT with AL in Low-Resource Settings | 2305.14576, EMNLP 2023 | adapters beat full fine-tuning inside AL | cite for LoRA |
| 7 | LLMaAA | 2310.19596, EMNLP Findings 2023 | LLM labels actively selected data, NER and RE | main baseline to cite |
| 8 | LLMs in the Loop | 2404.02261, ECML-PKDD 2024 | GPT-4 labels in AL for low-resource NER, 42x cheaper | related |
| 9 | ALLabel | 2509.07512, EMNLP 2025 | selects demonstrations for an LLM, not a student | partial overlap |
| 10 | Mixture of LLMs in the Loop | 2601.15773 | several small LLMs label; disagreement flags errors | teacher idea |
| 11 | HyPAC | 2602.02550 | routes to fast LLM, slow LLM or human with error bound | related to routing arm |
| 12 | ACT as Human | 2511.09833, NeurIPS 2025 | LLM labels all, humans review flagged cases | human-routing related |
| 13 | LLM on a Budget: Active Knowledge Distillation | 2511.11574 | uncertain samples to the teacher, up to 80% fewer samples | closest concept, classification only |
| 14 | EvoKD | 2403.06414, COLING 2024 | LLM generates examples at student weak points | related |
| 15 | Distilling LLMs for Clinical IE | 2501.00031 | distilled BERT beats teacher, 12x faster, up to 101x cheaper | supports student-beats-teacher; cost table format |
| 16 | Error-Type-Aware Loss Reweighting for Noisy LLM NER Labels | 2608.30827 | +0.8 to 2.0 F1 | optional noise add-on |
| 17 | GLiNER-BioMed; FiNERweb | 2504.00676; 2512.13884 | GLiNER-style students trained on LLM labels, no selection | LLM-to-GLiNER is standard |
| 18 | Conformal Prediction for NER | 2601.16999 | coverage-guaranteed label sets | calibration tool |
| 19 | Reliable Financial NER Under Domain Shift | 2608.19558 | span probability stays reliable under shift; 3 seeds + bootstrap | copy the evaluation practice |
| 20 | Data Augmentation for NER Uncertainty | 2407.02062 | span calibration metric | metric source |

Also from the Codex review: UCCI (2605.18796, calibrated cascades incl. NER); 2204.08491 (AL gains with pretraining); 2309.06131 (random competitive in neural ranking). Classic "simple baseline wins" AL papers: 1912.05361 ("Parting with Illusions about Deep Active Learning"), CVPR 2022 robust AL, EMNLP 2019 practical obstacles.

## What reviewers will raise

1. Random is as good. Answer: seeds, intervals, the pre-registered contrast.
2. Why not label the whole pool with the LLM? Answer: the full-pool baseline and the cost views.
3. The teacher is weak. Answer: Gemma 4 12B plus Qwen 3.8 27B, and a teacher error analysis.

## Changelog

- 2026-09-25 10:50 CEST - Created from the literature scan and Codex round 1 and 2 sources.
