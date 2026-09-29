---
title: Training playbook for the GLiNER2 student
date: 2026-09-25 23:35 CEST
author: Claude
type: wiki
status: draft
---

# Training playbook for the GLiNER2 student

This page turns the research pages into rules for our runs. Each rule names its evidence. The detail lives in:

+ `gliner2.md`: architecture, data formats, trainer, pitfalls in the installed code.
+ `gliner-core-papers.md`: GLiNER, GLiNER multi-task, GLiNER-BioMed and other follow-ups.
+ `relation-extraction.md`: GLiREL, entity-marker methods, CrossRE.
+ `classification-and-slots.md`: GLiClass, Hallmarks, MASSIVE.
+ `datasets-and-annotation-literature.md`: NER datasets, LLM annotation and active learning.

## 1. Rules for every task

| Rule | Why | Evidence |
|---|---|---|
| Put LoRA on `["encoder", "all_task_heads"]` or on the encoder plus the task head. Never tune only the heads. | Heads-only trials reached about 0.60 dev F1 on CleanCoNLL; encoder trials reached 0.87. Same pattern on Hallmarks. | Our tuning (`runs/tuning/*/trials.jsonl`); `gliner2.md` section on LoRA targets |
| Use one learning rate (`task_lr`). | With LoRA on, gliner2 ignores `encoder_lr`. Every adapter trains at `task_lr`. | `gliner2.md`; trainer.py:1352-1360 |
| Turn augmentation off. | The default augmentation renames labels and drops labels. With renamed labels it can train a true label as "no" (Codex round 31: 16 CrossRE examples). All three frozen recipes chose it off. | `gliner2.md` pitfalls; processor.py:1071-1093 |
| Train from exact offsets, not surface strings. | String targets mark every copy of the text (0.13% to 0.68% of pool spans changed). | Design section 17; `exact_targets.py` |
| Stop early and pick the checkpoint on dev micro-F1. | gliner2 stops on `eval_loss` by default. We set `metric_for_best="dev_f1"`, `greater_is_better=True`. | train.py:116-119; trainer.py:2025 |
| Choose the decision threshold on dev only. | The threshold moves F1 a lot. Rules tuned on dev can fail on test, so the test sweep stays descriptive. | `gliner-core-papers.md` (BioASQ); design section 17 |
| Keep sentences with no gold item in training. | gliner2 drops only `"entities": {}`. Our examples list every type with an empty list, so they stay (checked 25 Sep). | `gliner2.md` pitfalls; our check on 400 records per dataset |
| Run every trial and run in its own process. | GPU memory built up across trials in one process and caused out-of-memory failures. | Design section 17 |

## 2. Rules per task

### NER (CleanCoNLL, BC5CDR, MIT Movie)

+ Frozen recipe: r 32, alpha 64, `task_lr` 3.1e-4, batch 8, no augmentation (dev micro-F1 0.876).
+ Label descriptions exist in the data format (`entity_descriptions`) but the student does not use them yet. Test them on dev before any change.
+ The teacher produces nested spans on BC5CDR and MIT Movie. The validator keeps the first span and counts the rest as `overlap_conflict`.
+ Teacher labels on empty sentences are mostly false positives (NoiseBench: 45.4% of GPT-3.5 errors are made-up entities; our pool analysis agrees).

### Multi-label classification (Hallmarks)

+ Frozen recipe: r 32, alpha 64, encoder plus classification head, `task_lr` 4.0e-4, dropout 0.1, batch 8 (dev micro-F1 0.441).
+ Use `Classifier.batch_score`, not `classify_text`, because `classify_text` never returns an empty set.
+ Published BLURB scores (about 82 micro-F1) label whole abstracts. We label sentences, so they are not a fair yardstick.
+ Candidates for a dev test: label descriptions, and a class-weighted or focal loss (GLiClass uses focal loss, alpha 0.7). gliner2 has no loss weight per label now.

### Slots (MASSIVE en-US and fr-FR)

+ Frozen recipe: r 32, alpha 64, encoder plus extraction head, `task_lr` 3.5e-4, batch 16 (dev score 0.730).
+ Use `Structure("slots", mode=None)`. Record mode in `extract_json` is unsafe.
+ Expect fr-FR to score about 6 to 8 points below en-US (MASSIVE paper, Table 9).

### Relations over given mentions (CrossRE)

+ 88% of ordered pairs have no relation. Published given-entity scores that include no-relation pairs are low (8.6 to 19.0 macro-F1, Silver Syntax, Table 1).
+ Keep every positive pair and sample no-relation pairs with a dev-tuned ratio. Evaluate over all pairs.
+ The classification-head formulation learns slowly (train F1 0.575 on 16 sentences after about 20 epochs). The head reads only the label tokens, so the pair reaches it only through attention.
+ Two better formulations are being built (Codex round 32), to be chosen on dev:
  + `native`: the checkpoint's own relation scorer on the given mention offsets.
  + `marker`: pool the states at the typed `[H:` and `[T:` markers into a small 17-output head (Matching the Blanks, PURE, the CrossRE baseline).
+ Do not up-weight positives without dev evidence. GLiNER reports that extra positive weight hurts F1.

## 3. Checks before a long search

1. Memory test: train on 16 sentences and score the same 16. Expect F1 near 1.0. If not, fix the formulation before tuning.
2. One short dev run (600 steps). Check that the dev F1 curve rises.
3. Check the GPU memory log across two trials.

## 4. Open questions

+ Do label descriptions help the student on dev (NER, Hallmarks, slots)?
+ Which relation formulation wins on dev?
+ How large is the English gap between GLiNER2.5 multi and base on our datasets? The Fastino blog reports 3.45 F1 on zero-shot CrossNER.

## Changelog

- 2026-09-25 23:35 CEST - Created from the five research pages, the tuning logs and the round-31 code check.
