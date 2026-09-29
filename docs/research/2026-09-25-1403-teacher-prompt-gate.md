---
title: Teacher prompt gate, v1 vs v2 (Gemma 4 12B)
date: 2026-09-25 14:03 CEST
author: Claude (for Abhishek Thomas)
type: research
status: frozen
---

# Teacher prompt gate, v1 vs v2 (Gemma 4 12B)

Rule (design section 15): per dataset, compare prompt v1 (label names) and v2 (label names plus one short definition from `configs/label_definitions.yaml`) on the same first 200 dev sentences. Keep v2 only if dev micro-F1 is higher; ties keep v1. MASSIVE fr-FR inherits the en-US choice. The chosen prompt is used unchanged for every teacher.

Setup: Gemma 4 12B QAT UD-Q4_K_XL (sha256 `90fd44e2…`), llama.cpp commit 611dc03, 8 parallel slots, temperature 0, JSON-schema constrained output, thinking off. CrossRE uses the given-entity pair task (design section 15).

| Dataset | v1 micro-F1 | v2 micro-F1 | v1 macro-F1 | v2 macro-F1 | v2 valid rate | Chosen |
|---|---:|---:|---:|---:|---:|---|
| CleanCoNLL | 0.7801 | 0.7943 | 0.6750 | 0.7532 | 1.00 | v2 |
| BC5CDR | 0.8013 | 0.8424 | 0.7929 | 0.8344 | 1.00 | v2 |
| MIT Movie | 0.7067 | 0.7273 | 0.5805 | 0.5829 | 0.96 | v2 |
| CrossRE (given entities) | 0.1460 | 0.2689 | 0.1556 | 0.2862 | 0.94 | v2 |
| Hallmarks | 0.4530 | 0.4393 | 0.3415 | 0.3380 | 1.00 | v1 |
| MASSIVE en-US | 0.5012 | 0.5726 | 0.3250 | 0.3958 | 1.00 | v2 |
| MASSIVE fr-FR (inherits en-US) | 0.4328 | 0.4683 | 0.3297 | 0.3628 | 1.00 | v2 |

Pilot (before design section 15): end-to-end CrossRE triples, v1, 0.0710.

Notes:

+ These are 200-sentence dev estimates for a prompt choice, not paper results.
+ The MIT Movie v2 score (0.727) is close to the thesis Gemma 3 12B test score (69.4), which used other labels and dev equal to test.
+ Frozen in `configs/prompts.yaml`.

## Changelog

- 2026-09-25 14:03 CEST - Created.
