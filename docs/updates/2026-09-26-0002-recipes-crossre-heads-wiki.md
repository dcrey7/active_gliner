---
title: Three recipes frozen, CrossRE relation heads, literature wiki, first teacher-error evidence
date: 2026-09-26 00:02 CEST
author: Claude (review, tests, docs) with Codex (code)
type: update
status: frozen
---

# Three recipes frozen, CrossRE relation heads, literature wiki, first teacher-error evidence

## 1. What we did

+ Fixed the tuning memory leak for good: every trial and every matrix run starts in its own process (`58a4a3f`).
+ Ran the dev-only tuning searches for NER, classification and slots, and froze their recipes (`da3f8bc`, `8cb8d8d`, `8bd4bde`).
+ Found that CrossRE did not learn. Diagnosed it with memory tests, a code check (Codex round 31) and a literature study.
+ Added training-only negative sampling for CrossRE (`9866f89`) and three relation heads chosen on dev (`d1de051`, Codex round 32).
+ Wrote seven wiki pages from a full read of the GLiNER family and our datasets (`068120f`, `f654a14`, `5da6786`).
+ Measured teacher error against zero-shot student confidence on five pools (`f3e2080`).
+ Started the 20-trial CrossRE search that includes the relation head (log `tune_crossre2.log`).

## 2. What we measured

**Frozen recipes (dev-only, 20 trials each, no failed trial):**

| Task | Dataset | Dev score | LoRA targets | r / alpha | task_lr | Batch | Augmentation |
|---|---|---:|---|---|---:|---:|---|
| NER | CleanCoNLL | 0.876 | encoder + all heads | 32 / 64 | 3.1e-4 | 8 | off |
| Classification | Hallmarks | 0.441 | encoder + classification head | 32 / 64 | 4.0e-4 | 8 | off |
| Slots | MASSIVE en-US | 0.730 | encoder + extraction head | 32 / 64 | 3.5e-4 | 16 | off |

**CrossRE memory tests** (16 pool sentences, train and score the same 16, lr 3e-4):

| Head | 1:1 negatives, 400 steps: F1 | All negatives, 800 steps: P / R / F1 | Train time (all negatives) |
|---|---:|---|---:|
| classifier (inline markers, classification head) | 0.471 | 0.623 / 0.585 / 0.603 | 111 s |
| marker (typed-marker start pooling, 17-way head) | 0.614 | 0.917 / 0.677 / 0.779 | 76 s |
| native (checkpoint relation scorer on given offsets) | 0.632 | 0.921 / 0.892 / **0.906** | 281 s |

+ The 1:1 test is unfair: it scores 394 "no relation" pairs the model never saw.
+ The first classifier search (before negative sampling) reached dev micro-F1 0.0 to 0.035 in five trials.

**Teacher labelling (Gemma 4 12B, pool):**

| Pool | Valid rate | Main error kind |
|---|---:|---|
| CleanCoNLL | 99.95% | not in text (7), transport (16, retried later) |
| BC5CDR | 97.8% | overlap conflict (141) |
| MIT Movie | 94.9% | overlap conflict (395) |
| CrossRE | 94.9% | bad mention (93), JSON (66) |
| Hallmarks | 100% | none |
| MASSIVE en-US | done | see `label_all2.log` |

**Teacher error by zero-shot student confidence:** see `docs/research/2026-09-25-2328-teacher-vs-uncertainty.md`. On BC5CDR, teacher F1 is 0.28 in the least-sure decile and 0.89 in the most-sure decile. The least-sure NER deciles are mostly sentences with no gold entity.

## 3. What failed or stays unknown

+ Real error before the process fix: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 734.00 MiB ... this process has 18.79 GiB memory in use.`
+ `free_gpu()` alone did not stop the leak (2.84, 3.96, 5.07 GiB after trials 0 to 2). Process isolation did.
+ I started three GPU memory tests at once next to the 11 GiB teacher. One ran out of memory. They now run one after another.
+ I misread PURE Table 7 once (pipeline against joint, not markers against no markers). The wiki page is corrected.
+ CrossRE has no `source.json`, so runs record no CrossRE dataset revision (research rule 7). Codex must add it.
+ The 50 GPU model tests have not run since the round-27 changes.
+ gliner2 bug (not ours, our recipes avoid it): with augmentation on, renamed labels can train a true label as "no" (processor.py:1071-1093).

## 4. What changed on disk

+ Code: `src/active_gliner/{tune.py, tune_trial.py, process.py, matrix.py, matrix_run.py, train.py, run.py, model.py, protocol.py, byod.py}`, `src/active_gliner/tasks/{relations.py, relation_heads.py, relation_training.py}`, `scripts/relation_memory_test.py`, `justfile`.
+ Recipes: `configs/recipes/{ner, classification, slots}.yaml`.
+ Tests: `tests/test_protocol_fixes.py`, `tests/test_relations_pairs.py`, `tests/test_relation_heads.py`. `just check-cpu`: 156 passed, 50 deselected.
+ Docs: design section 17; `docs/wiki/{gliner-core-papers, gliner2, relation-extraction, classification-and-slots, datasets-and-annotation-literature, training-playbook, gli-family-tree}.md`; `docs/wiki/assets/gli-family-tree-2026-09.jpg` (internal only, credit Ihor Stepanov); `docs/research/2026-09-25-2328-teacher-vs-uncertainty.*`.
+ Archives: `runs/tuning_archive/{cleanconll-old-targets, hallmarks-aborted, cleanconll-leak-run3, crossre-imbalanced}`.

## 5. What the next session must do first

1. Read `tune_crossre2.log` and `runs/tuning/crossre/trials.jsonl`. Check for FAIL rows and that `configs/recipes/relations.yaml` exists with `relation_head`.
2. Ask Codex to write CrossRE `source.json` (dataset revision) so pins are complete.
3. When labelling ends, rerun `label_all.py` once to retry the 16 CleanCoNLL transport failures.
4. Run `just check` with the GPU model tests, then the O7 timing gate and the same-seed check.
5. Start `active-gliner matrix run`, ground-truth blocks first.

## Changelog

- 2026-09-26 00:02 CEST - Created.
