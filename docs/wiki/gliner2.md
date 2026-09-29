---
title: GLiNER2 and Fastino models
date: 2026-09-25 23:25 CEST
author: Claude (research agent)
type: wiki
status: draft
---

# GLiNER2 and Fastino models

This page is the reference for the student model family. It covers the GLiNER2 paper, the `gliner2` package, the Fastino model cards and the pitfalls we found in the code.

Conventions on this page:

+ `P/` means the installed package: `.venv/lib/python3.11/site-packages/gliner2/` (version 2.0.0, PyPI upload 2026-08-24).
+ "README" means the package README in `.venv/lib/python3.11/site-packages/gliner2-2.0.0.dist-info/METADATA`.
+ "Tutorial N" means `tutorial/N-*.md` in https://github.com/fastino-ai/GLiNER2 (main branch, read 2026-09-25).
+ "Paper" means arXiv 2507.18546 v1 (24 Jul 2025), also EMNLP 2025 System Demonstrations, pages 130 to 140 (https://aclanthology.org/2025.emnlp-demos.10/). Tables 2 to 6 are the same in both versions.
+ "Card" means the Hugging Face model card. The local copy of the multi card is `~/.cache/huggingface/hub/models--fastino--gliner2.5-multi-v1/snapshots/12fc40399dae672ce840c5e3c50a92340bff3c8c/README.md`.
+ Earlier study with tested runs: `docs/research/2026-09-25-1202-gliner2-api-study.md`.

## 1. Two architectures, one package

The package holds two model types behind one API.

| Name | Class | Candidate search | Checkpoints | Source |
|---|---|---|---|---|
| GLiNER2 ("span") | `GLiNER2` / `SpanExtractor` | every span up to `max_width` words (8) | `gliner2-base-v1`, `gliner2-large-v1`, `gliner2-multi-v1` | Paper; Hub `config.json` (`"max_width": 8`) |
| GLiNER2.5 ("boundary") | `BoundaryExtractor` | sparse start/end pairs, any length inside the window | `gliner2.5-small-v1`, `gliner2.5-base-v1`, `gliner2.5-multi-v1` | README "Architecture guide"; card "Model details" |

+ `AutoExtractor.from_pretrained` reads `architecture` from `config.json` and picks the class (P/auto.py:91-131).
+ `GLiNER2.from_pretrained` is the legacy span loader. It does not load GLiNER2.5 checkpoints (card "Load the model").
+ The paper describes only the span model. GLiNER2.5 has no paper. Its only technical source is a Fastino blog post (section 5.3).

### Model table

| Model | Params | Encoder | Language | Source |
|---|---|---|---|---|
| gliner2-base-v1 | 205M | deberta-v3-base | English | README "Available Models" |
| gliner2-large-v1 | 340M | deberta-v3-large | English (Hub tags en, fr, es) | README; Hub tags |
| gliner2-multi-v1 | about 205M | mdeberta-v3-base | Multilingual (Hub tags fr, en, es, de, it, pt) | README; Hub tags |
| gliner2.5-small-v1 | 74M | deberta-v3-xsmall | English | card |
| gliner2.5-base-v1 | 194M | deberta-v3-base | English | card |
| gliner2.5-multi-v1 | 287M, about 594 MB, mostly FP16 | mdeberta-v3-base | Multilingual, no language list | card "Model details" |

+ All GLiNER2.5 cards are Apache-2.0. Their `config.json` files differ only in `model_name` (checked on the local snapshots).
+ GLiNER2.5 config: `max_len 4096`, `token_pooling "first"`, `enable_relations true`, `enable_records true`, `enable_count_head true`, `enable_abstention true`, `overlap_policy "flat"`, `candidate_budget 192`, `pool_size 192`, `candidate_pool "shared"`, `max_gold_per_query 64`, `transformers_version "5.8.0"` (local `config.json`).
+ The encoder config of the multi model has `vocab_size 250112`, 12 layers, hidden size 768, `max_position_embeddings 512`, relative attention (local `encoder_config/config.json`).
+ No GLiNER2.5 large exists. Newer checkpoints on the Hub: `GLiNER2.5-Decide` (span, deberta-v3-large, 340M), `GLiNER2.5-multi-Decide` (boundary, base `gliner2.5-multi-v1`), `GLiNER2.5-Decide-1B`. These are decision (classification) fine-tunes (Hub cards, created 23 Sep 2026).
+ Both multi snapshots in our cache (`12fc4039…` and `a221b77a…`) point to the same blobs for config, weights and tokenizer.

## 2. Architecture

### 2.1 The schema prefix and special tokens

In simple words: the model reads the task description and the text in one input. Special marker tokens stand for each label. The contextual vector of each marker becomes the "query" for that label.

Example for NER with two labels:

```
( [P] entities ( [E] person [E] location ) ) [SEP_TEXT] john lives in paris .
     |              |            |
  task marker    query 1      query 2  ----> scored against spans of the text
```

The package defines ten special tokens (P/processor.py:284-298):

| Token | Role |
|---|---|
| `[P]` | start of one task; the task name (and prompt, descriptions, examples) follows |
| `[E]` | one entity type (NER) |
| `[C]` | one field of a JSON structure |
| `[R]` | one role of a relation (for example head, tail) |
| `[L]` | one classification label |
| `[SEP_STRUCT]` | separates two task segments |
| `[SEP_TEXT]` | separates the schema from the text |
| `[DESCRIPTION]` | a label description inside the task string: `[DESCRIPTION] label: text` |
| `[EXAMPLE]`, `[OUTPUT]` | few-shot pairs for classification |

+ One segment is `( [P] <name>[: prompt][ [DESCRIPTION] l: d …][ [EXAMPLE] in [OUTPUT] out …] ( <marker> f1 <marker> f2 … ) )` (P/processor.py:1096-1131).
+ Segments join with `[SEP_STRUCT]`; `[SEP_TEXT]` comes before the text (P/processor.py:1253-1261).
+ Only the `[P]` slot and the child marker slots are routed as query vectors (P/processor.py:1263-1275).
+ The tokenizer adds the tokens as special tokens (P/processor.py:318-320). The multi tokenizer lists them in `extra_special_tokens` (local `tokenizer_config.json`).
+ The paper (Appendix A) lists `[P]`, `[E]`, `[C]`, `[L]` and `[SEP]`. It does not mention `[R]`, `[DESCRIPTION]` or the split into `[SEP_STRUCT]` and `[SEP_TEXT]`. It says the tokens are "randomly initialized and learned during training".
+ Text words: the whitespace splitter lower-cases every word token (P/processing/word_splitter.py:34-42; called with `lower=True` at P/processor.py:571).
+ The collator adds a `.` to text that does not end in `.`, `!` or `?` (P/processor.py:521-525).
+ Word vectors use the first subword (`token_pooling "first"`).

### 2.2 Span model (GLiNER2, the paper)

| Task | Scoring | Loss (code) |
|---|---|---|
| Entities | Paper Eq. 1: `score(s, e) = sim(h_s, h_e)`, dot product, sigmoid, keep above 0.5. Spans up to `max_width` 8 | BCE over an (instance x field x start x width) grid, with 50% of negative cells masked at random in training (P/models/span/model.py:571-632) |
| Structures | An MLP on `[P]` predicts the instance count K (20 classes, 0 to 19). `[C]` vectors are conditioned on K occurrence ids, then scored like entities | same grid BCE, plus count cross-entropy (P/models/span/model.py:404-413); count is capped at 19 (line 406, 592) |
| Classification | Paper Eq. 2: `logit_i = MLP(h_l_i)`; softmax for single-label, sigmoid for multi-label | BCE with sum reduction on the `[L]` logits (P/models/span/model.py:380-387) |
| Relations | not in the paper; coded as a structure with `[R]` roles | as structures |

+ Total loss = classification + structure + count (P/models/span/model.py:271-275).
+ The paper does not report the losses.

### 2.3 Boundary model (GLiNER2.5, our student)

In simple words: the model does not list every span. It predicts, for each label query, which words can start a span and which can end one. It pairs the best starts and ends into a short list of candidates. A second scorer (the "reranker") scores each candidate for the query.

```
text words ──> encoder ──> boundary encoder ──> start / end / inside scores per query
                                                   │
                         shared pool of up to 192 candidate spans (start,end)
                                                   │
                         pair scorer (query x candidate) ──> sigmoid ──> threshold
```

Parts of `BoundaryExtractorModel` (P/models/boundary/model.py:1117-1198):

+ `encoder` - the mDeBERTa encoder.
+ `classifier` - MLP hidden to 2 x hidden to 1, ReLU, applied to each `[L]` vector (lines 1156-1163).
+ `boundary_head` - boundary encoder, query head (start, end, inside marginals), proposer, pair scorer, shared pool builder and scorer, abstention gate (`null_projection`), count head (lines 149-255).
+ `record_decoder` - the record head, only if `enable_records` (lines 1173-1179).
+ `relation_pair_generator` and `relation_scorer` - only if `enable_relations` (lines 1180-1198).

How each task is scored:

| Task | Query | Scoring | Source |
|---|---|---|---|
| Entities | each `[E]` vector | boundary proposals, then pair logits; sigmoid over `pair_temperature` | P/models/boundary/engine.py:78-94 |
| Structures | each `[C]` vector | same as entities; legacy decode puts all fields in one instance; record mode uses the record head | P/models/boundary/engine.py:487-592 |
| Classification | each `[L]` vector | `classifier` MLP gives one logit per label | P/models/boundary/model.py:1594-1636 |
| Relations | the two `[R]` vectors of one relation | head and tail are boundary queries too; relation state = concat(head marker, tail marker); pairs = top head candidates x top tail candidates; `SparseRelationScorer` MLP on four endpoint states, relation state, order and distance, plus a biaffine content term | P/models/boundary/model.py:1480-1497; P/models/boundary/relations.py:155-233, 357-410 |

Boundary losses and weights (P/models/boundary/model.py:674-946; weights from the checkpoint `config.json`):

| Loss | What it trains | Weight |
|---|---|---|
| start, end | start and end marginals; asymmetric focal (gamma_neg 2, gamma_pos 0, clip 0.05), negatives weighted 0.5 | 1.0 each (`DEFAULT_LOSS_WEIGHTS`, line 95) |
| pair | BCE over candidates, all positives plus 20 hard negatives per positive (at least 16) | 1.0 |
| inside | inside-span consistency | 0.5 |
| soft IoU | soft targets by overlap with gold | 0.2, annealed to 0 over 20,000 steps |
| rerank listwise | gold mass over reranked candidates | 0.3 |
| proposal listwise | ranks gold above other proposals | 0.3 |
| consistency | marginals match candidate noisy-OR | 0.1, warmed up over 2,000 steps |
| abstention | per-query gate, target 1 when the label is absent | 0.2 |
| count | Poisson NLL on the number of gold mentions | 0.2 |
| classification | BCE per label, divided by the label count | `classification_loss_weight` 1.0 |
| relation | BCE over proposed pairs, mean over valid pairs | `relation_loss_weight` 1.0 |
| record | object loss + field loss | `record_loss_weight` 1.0 |

+ Total = boundary total + classification + record + relation (P/models/boundary/model.py:1863-1874).
+ Absent queries: in training, the pair loss keeps all queries with a gold span and a random sample of absent queries, `negative_query_ratio` 1.0 times the positive count, at most 64 (lines 784-800).
+ Gold injection: in training, gold spans are added to the proposals with a probability that starts at 1.0, holds for 15% of the planned steps, then falls linearly to 0.25 (P/training/trainer.py:1093-1103, 1131-1140; config defaults lines 239-241). Evaluation never injects gold (P/models/boundary/model.py:405).
+ Relation gold labels match proposed pairs by exact coordinates (P/models/boundary/model.py:1715-1738). A gold pair that is not in the proposals gives no positive signal.

## 3. Training data formats

Each JSONL line is `{"input": text, "output": {...}}`. The form `{"text": ..., "schema": {...}}` also works (P/training/data.py:280-313; P/training/trainer.py:473-479). Valid output keys: `entities`, `entity_descriptions`, `classifications`, `json_structures`, `json_descriptions`, `record_metadata`, `relations` (Tutorial 8; P/training/data.py:917-943).

### 3.1 Entities

```json
{"input": "Dr. Sarah Johnson prescribed Metformin 500mg for diabetes.",
 "output": {"entities": {"person": ["Dr. Sarah Johnson"], "medication": ["Metformin"], "condition": ["diabetes"]},
            "entity_descriptions": {"medication": "Names of drugs or pharmaceutical products"}}}
{"input": "The conference will be held next week.",
 "output": {"entities": {"person": [], "organization": [], "location": []}}}
```

+ The processor builds one instance: `[1, [[mentions of type 1], [mentions of type 2], …]]` (P/processor.py:969-971).
+ An empty list is a negative label: the query gets no gold span.
+ Descriptions go into the prompt as `[DESCRIPTION] label: text` (P/processor.py:1111-1116).

### 3.2 Classifications

```json
{"input": "This smartphone has an amazing camera but the battery life is poor.",
 "output": {"classifications": [{"task": "product_aspects",
   "labels": ["camera", "battery", "screen", "performance", "design"],
   "true_label": ["camera", "battery"], "multi_label": true,
   "label_descriptions": {"camera": "Photo and video quality"},
   "prompt": "Which aspects does the review mention?"}]}}
```

+ Targets are one 0/1 value per label: 1 if the label is in `true_label` (P/processor.py:1197).
+ `true_label: []` gives all zeros, which is a valid negative.
+ `multi_label` does not change training. The loss is BCE per label in both cases (P/models/boundary/model.py:1629). `multi_label` changes only decoding (P/inference/runtime.py:403-424).
+ More than one true label sets `multi_label=True` automatically (P/training/data.py:413-415).
+ `examples` are `[input, output]` few-shot pairs (Tutorial 8).

### 3.3 Structures (json_structures)

```json
{"input": "Book a single room at Grand Hotel for 2 nights.",
 "output": {"json_structures": [{"booking": {"hotel": "Grand Hotel", "nights": "2",
   "room_type": {"value": "single", "choices": ["single", "double", "suite"]}}}]}}
```

+ Field values: a string, a list of strings, `""` or `null` for absent, or a choice field `{"value", "choices"}` (P/training/data.py:486-491, 976-977).
+ Choice fields are written into a prefix before the text, `( booking: room_type ( single | double | suite ) )`, and the target is the choice word in that prefix (P/processor.py:748-806, 1158-1176).
+ Several instances of one structure become `[count, [instance, instance, …]]` after deduplication (P/processor.py:906-927).
+ Modes (record metadata): `natural` (anchor field seeds each record), `latent`, `anchorless`; `None` means the legacy aggregate decoder (P/processing/boundary_preprocessing.py:211-215; P/training/data.py:502-546).
+ `Structure(...)` defaults to `mode="natural"` with the first field as anchor (P/training/data.py:507, 538-541). See pitfall 4 in section 7.
+ Cardinality: `optional_one`, `required_one` (never absent), `zero_or_more`, `one_or_more` (P/processing/records.py:36-55).

### 3.4 Relations

```json
{"input": "John works for Apple Inc. and lives in San Francisco.",
 "output": {"relations": [{"works_for": {"head": "John", "tail": "Apple Inc."}},
                          {"lives_in": {"head": "John", "tail": "San Francisco"}}]}}
```

+ Field names other than head and tail are allowed, for example `{"transaction": {"sender": …, "recipient": …, "amount": …}}` (Tutorial 8).
+ The first occurrence of a relation type fixes its field names (Tutorial 8; P/training/data.py:752-760).
+ The schema is `( [P] works_for ( [R] head [R] tail ) )` (P/processor.py:1022-1027).
+ Head and tail spans become gold mentions for the two role queries (P/processing/boundary_preprocessing.py:446-453).
+ Gold pairs are all head occurrences times all tail occurrences (P/processing/boundary_preprocessing.py:407-420).

### 3.5 How the processor turns strings into targets

In simple words: you give the label as a string. The processor searches the text for that string, word by word. It marks every place where the string appears.

Example: text "Paris is nice. I love Paris." with `location: ["Paris"]` gives two gold spans, one for each "Paris".

+ `_find_sublist` returns all matches of the word sequence (P/processor.py:1206-1237).
+ Matching runs on lower-cased word tokens, so it ignores case (P/processor.py:1239-1241, 571).
+ A string that is not found gives `(-1, -1)`. For entities this raises `ValueError` in training, unless `allow_invalid_samples=True` (P/processing/boundary_preprocessing.py:430-441). For structure and relation fields it is skipped (lines 446-453).
+ Our fix (design section, round 27): our code marks only the supplied offsets. Before the fix, 0.13% to 0.68% of pool spans matched a non-gold copy (`docs/updates/2026-09-25-1533-round27-28-fixes-tuning-restart.md`).

## 4. Training

### 4.1 TrainingConfig fields and defaults

Source: P/training/trainer.py:180-270.

| Group | Defaults |
|---|---|
| Length | `num_epochs=10`, `max_steps=-1` (then epochs decide) |
| Batch | `batch_size=2`, `eval_batch_size=8`, `gradient_accumulation_steps=1` |
| Learning rates | `encoder_lr=1e-5`, `task_lr=5e-4` |
| Optimizer | AdamW, `weight_decay=0.01`, betas 0.9 and 0.999, eps 1e-8, `max_grad_norm=1.0`, fused on CUDA |
| Schedule | `scheduler_type="linear"` (also cosine, cosine_restarts, constant), `warmup_ratio=0.1`, `warmup_steps=0` (overrides the ratio when above 0) |
| Precision | `fp16=True` when unset; boundary models switch to bf16 when you set neither (lines 694-705) |
| Evaluation | `eval_strategy="steps"`, `eval_steps=500`, `metric_for_best="eval_loss"`, `greater_is_better=False`, `save_best=True`, `save_total_limit=3` |
| Early stopping | `early_stopping=False`, `early_stopping_patience=3`, `early_stopping_threshold=0.0` |
| Data | `validate_data=True`, `max_len=None`, `group_by_length=True`, `num_workers=4`, `seed=42` |
| Boundary | `gold_injection_start=1.0`, `gold_injection_end=0.25`, `gold_injection_hold_frac=0.15`, `on_capacity_exceeded="raise"`, `strict_training=True`, `allow_invalid_samples=False` |
| LoRA | `use_lora=False`, `lora_r=16`, `lora_alpha=32.0`, `lora_dropout=0.0`, `lora_use_dora=False`, `lora_target_modules=["encoder","span_rep","classifier","count_embed","count_pred"]`, `save_adapter_only=True` |

Tutorial 9 shows `batch_size=32`, `eval_batch_size=64` and `logging_steps=50` in its "Complete Configuration Reference". The code defaults are 2, 8 and 1.

### 4.2 task_lr and encoder_lr

| Mode | Parameter groups | Source |
|---|---|---|
| No LoRA | encoder parameters (name contains "encoder") at `encoder_lr`; all other parameters at `task_lr` | P/training/trainer.py:1361-1384 |
| LoRA | all parameters frozen, then only LoRA A/B matrices train, all at `task_lr`; `encoder_lr` is ignored | P/training/trainer.py:855-857, 1352-1360 |

So with LoRA, `task_lr` is the learning rate of the encoder adapters too.

### 4.3 LoRA targets

+ LoRA wraps only `nn.Linear` layers (P/training/lora_targets.py:66-89).
+ `"encoder"` selects encoder Linear layers whose names contain query, key, value or dense (lines 11, 72-77).
+ `"encoder.query"` and similar select one kind (lines 78-83).
+ Aliases (lines 34-53): `all_task_heads` (all heads of the model), `extractive_head` (`boundary_head`), `classification_head` (`classifier`), `relation_head` (`relation_scorer`), `record_head` (`record_decoder`).
+ Boundary head names: `classifier`, `boundary_head`, plus `record_decoder` and `relation_scorer` when enabled (P/models/boundary/model.py:1090-1097).
+ The default list names span-model heads. On a boundary model it selects only the encoder and `classifier`. The boundary head, relation scorer and record head stay frozen. The API study counted 74 wrapped layers for the default and 131 for `["encoder", "all_task_heads"]` (`docs/research/2026-09-25-1202-gliner2-api-study.md`).
+ LayerNorms and other non-Linear weights never train under LoRA.
+ Checkpoints with `save_adapter_only=True` are PEFT-native (`adapter_config.json`, `adapter_model.safetensors`) (P/training/trainer.py:2101-2119). Load with `PeftModel.from_pretrained(base, path)` (P/training/trainer.py:2212-2218).

### 4.4 Default data augmentation

Training uses `SamplingConfig` (P/processor.py:241-262). Inference uses none (P/processor.py:814).

| Task | Augmentation | Probability | Source |
|---|---|---|---|
| All | shuffle the order of task segments | always | P/processor.py:828-834 |
| Entities | rename types to `entity 1`, `entity 2`, … | 0.2 | lines 951-960 |
| Entities | drop the whole task / drop one type / shuffle types | 0.0 / 0.0 / off | lines 249-251 |
| Entities | drop descriptions (mode "none" vs "descriptions") | 0.5 when not renamed | lines 948, 973-975 |
| Structures | drop a whole structure | 0.2 | line 857 |
| Structures | shuffle fields | always | line 878 |
| Structures | drop a field | 0.2, not for record-mode structures | lines 881-886 |
| Structures | rename fields to `field 1`, … | 0.2, not for record mode | lines 895-903 |
| Relations | drop one relation instance | 0.2 | line 990 |
| Relations | swap the order of the head and tail role markers | 0.2 | lines 998-1002 |
| Classification | rename labels to `label 1`, … (description falls back to the real name) | 0.5 | lines 1046-1055 |
| Classification | drop a fraction of labels, fraction = Beta(1,1) x 0.5 | always | lines 1060-1064 |
| Classification | add the true label back | 0.5 | lines 1071-1078 |
| Classification | shuffle labels | always | lines 1080-1081 |
| Classification | prompt mode from few_shot, descriptions, both, none | uniform | lines 1044, 1057 |
| Choice fields | shuffle choices and choice fields | always | lines 759-766 |

You can pass your own `SamplingConfig` to `SchemaTransformer(sampling_config=...)` (P/processor.py:300-314).

### 4.5 Loss weights

+ `classification_loss_weight` (1.0) scales the classification BCE mean (P/models/boundary/model.py:1635-1636).
+ `relation_loss_weight` (1.0) scales the relation BCE (line 1738).
+ Both live in `boundary_head` in `config.json`. They are not `TrainingConfig` fields.
+ Boundary loss reduction is `"sum"` in the checkpoint config.

### 4.6 Evaluation, early stopping and checkpoints

+ Built-in evaluation is loss only: `eval_loss` plus proposal recall ratios (P/training/trainer.py:1995-2003). Pass `compute_metrics(model, eval_dataset)` for F1 (line 2005-2006).
+ The best metric is `metric_for_best` (default `eval_loss`) (lines 2011-2021).
+ Early stopping counts evaluations with no gain above `early_stopping_threshold`; it stops at `early_stopping_patience` (lines 2025-2038). It needs eval data (lines 1068-1075).
+ Folders: `checkpoint-<step>`, `best`, `final` (lines 1852, 2019-2020, 1899-1900). The trainer does not reload `best` at the end.
+ Eval data is not validated or sanitized (line 1339).

### 4.7 Recommended settings from the sources

| Source | lr | Epochs | Batch | Warmup | LoRA |
|---|---|---|---|---|---|
| Paper Table 5 (pretraining, span model) | backbone 1e-5, task layers 2e-5 | 5 | not reported | 1,000 steps, linear | none |
| README Quick Start | encoder 1e-5, task 5e-4 | 10 | 8 | default 0.1 | none |
| README LoRA example | task 5e-4 | 10 | 8 | default | r 8, alpha 16, dropout 0, targets `["encoder","all_task_heads"]` (written as `lora_targets`, see pitfall 10 in section 7) |
| README complete example | encoder 1e-5, task 5e-4, cosine | 15 | 16 | 0.1 | none; early stopping patience 3 |
| Tutorial 9 full fine-tune | encoder 1e-6 to 5e-5 (typical 1e-5), task 1e-4 to 1e-3 (typical 5e-4) | | | | |
| Tutorial 9 domain example | encoder 5e-6, task 1e-4 | 20 | 16 | 0.05 | |
| Tutorial 9 LoRA best practice | task 1e-4 to 1e-3 (typical 5e-4) | | | | start r 16, alpha 32, dropout 0.1; attention layers first |
| Tutorial 10 by data size | task 5e-4 | under 1K: 10; 1K to 10K: 5; over 10K: 3 | 8, grad acc 2 | | under 1K: r 4, alpha 8; 1K to 10K: r 8, alpha 16; over 10K: r 16, alpha 32; start with `["encoder"]` |

+ The sources do not agree on LoRA rank or targets.
+ Tutorial 10 says to add "100-1000+ examples for best results".
+ Per-task advice (Tutorial 1): classification thresholds 0.3 to 0.5 for multi-label, 0.5 to 0.7 for single-label; give label descriptions.
+ Tutorial 12: long text chunks of 384 words with 64 overlap; raise overlap to 96 to 128 for relations.
+ Our recipes come from dev-only Optuna searches, not from these tables (design section 12; commits `da3f8bc`, `8cb8d8d`, `8bd4bde`).

## 5. Inference

### 5.1 Main options

| Option | Default | Effect | Source |
|---|---|---|---|
| `threshold` | 0.5 | global cut on sigmoid pair probability | P/inference/runtime.py:1159 |
| per-label `threshold` | none | entity and structure field thresholds per query | P/models/boundary/engine.py:198-226 |
| `include_confidence` | False | adds `confidence` | P/models/boundary/engine.py:297-376 |
| `include_spans` | False | adds half-open character `start`, `end`; `text[start:end] == text` | card "Entity extraction" |
| `overlap_policy` | checkpoint `flat` | `allow`, `nested`, `flat`/`disallow` (max-total-score non-overlapping set), `longest` | P/inference/overlap.py:16-27, 60-199; P/inference/runtime.py:224-238 |
| `max_len` | None | truncates words; use `*_long` helpers for long text | P/processor.py:576-579; card "Long documents" |

+ Overlap is resolved per label, not across labels (P/models/boundary/engine.py:280-282).
+ Abstention: if a label's null probability is above `abstention_threshold` 0.5, that label outputs nothing (P/models/boundary/engine.py:268-277).
+ A span outside the candidate pool has no score (pool 192, API study section 7).

### 5.2 Classification

+ `classify_text`: single-label uses softmax and argmax; multi-label uses sigmoid and `cls_threshold`, but returns the argmax label when no label passes (P/inference/runtime.py:403-424).
+ `Classifier.batch_score(texts, schema)` returns `ClassificationScores` with the raw logit per label (P/classification/engine.py:141-146; P/classification/scoring.py:150-217). `probability()` applies temperature and sigmoid or softmax (scoring.py:96-107). This gives true per-label probabilities with no argmax fallback.
+ `Classifier.classify` adds constrained decoding (`exact`, `beam`, `independent`) and `ClassificationConfig` (P/classification/engine.py:35-64).

### 5.3 Relations

+ `extract_relations(text, types)` returns `{"relation_extraction": {type: [...]}}` (P/inference/runtime.py:1581-1590).
+ Pairs come only from the model's own entity candidates: top 32 heads and top 32 tails per relation, capped at `relation_pair_cap` 64 in the checkpoint (P/models/boundary/relations.py:155-233; config).
+ The head and the tail carry the same number, the pair score (P/models/boundary/engine.py:886-892).
+ Decoding merges edges with the same head text and tail text, and keeps the closest pair (P/models/boundary/engine.py:947-968). Two copies of one name give one edge.
+ Independent decoding does not check entity types (card "Relation extraction").

**Can relations be scored between given mentions?** Not through a public API.

+ `extract_relations` and `JointIE` both take pairs from boundary proposals (P/models/boundary/engine.py:815-831; P/joint_ie/engine.py:18-38 has no mention input).
+ `SparseRelationScorer.forward` scores any `RelationPairBatch` of (head, tail) coordinates (P/models/boundary/relations.py:328-410). Calling it with hand-built pairs is possible but private and untested.
+ Our design instead scores each ordered mention pair as one multi-label classification with marked head and tail, through `Classifier.batch_score` (design section "CrossRE").

### 5.4 JointIE

+ `JointIE` scores mention and relation candidates, then searches a consistent typed graph with constraints (unique head, no self loops) (card "Joint information extraction").
+ `JointIEConfig`: `optimizer="beam"`, `beam_size=32`, `candidate_threshold=0.05`, `relation_role_threshold=0.05`, `top_k_entities=32`, `top_k_roles=12`, `relation_pair_cap=128` (P/joint_ie/engine.py:18-38).
+ Check `result.feasible`; False means the hard constraints failed (card).

### 5.5 Structures

+ `extract_json` on a checkpoint with a record head uses record mode `natural`, first field as anchor, `str` fields `required_one`, list fields `zero_or_more` (P/inference/runtime.py:1546-1579).
+ `Schema.structure(name)` with `mode=None` gives the legacy decoder: one instance with all fields (P/models/boundary/engine.py:506-592).

## 6. Pretraining data and reported results

### 6.1 Paper pretraining data (Section 3.1, Appendix B.1, Table 6)

| Part | Examples | Notes |
|---|---:|---|
| Total | 254,334 | |
| Real text, annotated by GPT-4o | 135,698 | News 74,456; Law 19,798; Wikipedia 17,909; PubMed 16,400; ArXiv 7,135 |
| Synthetic, generated by GPT-4o | 118,636 | emails, messages, resumes, social posts, orders, banking records, sports commentary |

Languages: not reported.

### 6.2 Paper results

Table 2, zero-shot classification accuracy:

| Dataset | Labels | GPT-4o | GLiClass | DeBERTa-v3 | GLiNER2 |
|---|---:|---:|---:|---:|---:|
| SNIPS | 7 | 0.97 | 0.80 | 0.77 | 0.83 |
| Banking77 | 77 | 0.78 | 0.21 | 0.42 | 0.70 |
| Amazon Intent | 31 | 0.72 | 0.51 | 0.59 | 0.53 |
| SST-2 | 2 | 0.94 | 0.90 | 0.92 | 0.86 |
| IMDB | 2 | 0.95 | 0.92 | 0.89 | 0.87 |
| AG News | 4 | 0.85 | 0.68 | 0.68 | 0.74 |
| 20 Newsgroups | 20 | 0.68 | 0.36 | 0.54 | 0.49 |
| Average | | 0.84 | 0.63 | 0.69 | 0.72 |

Table 3, CrossNER zero-shot F1:

| Domain | GPT-4o | GLiNER-M | GLiNER2 |
|---|---:|---:|---:|
| AI | 0.547 | 0.518 | 0.526 |
| Literature | 0.561 | 0.597 | 0.564 |
| Music | 0.736 | 0.694 | 0.632 |
| Politics | 0.632 | 0.686 | 0.679 |
| Science | 0.518 | 0.581 | 0.547 |
| Average | 0.599 | 0.615 | 0.590 |

Table 4, CPU latency in ms by label count (GPT-4o through the API):

| Labels | GPT-4o | DeBERTa | GLiClass | GLiNER2 |
|---:|---:|---:|---:|---:|
| 5 | 358 | 1714 | 137 | 130 |
| 10 | 382 | 3404 | 131 | 132 |
| 20 | 425 | 6758 | 140 | 163 |
| 50 | 463 | 16897 | 190 | 208 |

+ CPU hardware: not reported.
+ Structure and relation results: not reported. Section 3.2 says no zero-shot benchmark exists for structures.
+ The paper has no Limitations section.
+ Table 1 gives 205M parameters and context 2,048 tokens for GLiNER2.

### 6.3 GLiNER2.5 results (Fastino blog, 24 Aug 2026)

Source: https://fastino.ai/blog/gliner2-5-span-free-information-extraction. The model cards have no benchmark tables. Training data: made by the "Fastino Data Agent"; size and teacher not reported. Training used sequences up to 4,096 words.

Zero-shot macro F1 on 16 datasets:

| Group | 2.5 Multi | 2.5 Base | GLiNER2 Multi | GLiNER2 Base |
|---|---:|---:|---:|---:|
| Overall | 56.17 | 54.87 | 56.09 | 53.34 |
| Classification | 72.44 | 69.86 | 70.32 | 68.89 |
| Extraction | 46.40 | 45.88 | 47.56 | 44.01 |

| Dataset | 2.5 Multi | 2.5 Base | GLiNER2 Multi | GLiNER2 Base |
|---|---:|---:|---:|---:|
| ag_news | 70.99 | 69.71 | 72.93 | 70.54 |
| clinc_oos | 61.32 | 62.20 | 62.59 | 63.62 |
| imdb | 85.96 | 88.10 | 89.42 | 89.70 |
| multilingual_sentiment | 79.42 | 63.14 | 81.30 | 57.57 |
| rotten_tomatoes | 74.67 | 81.51 | 78.11 | 82.91 |
| xnli | 62.30 | 54.49 | 37.55 | 49.01 |
| crossner_ai | 45.60 | 50.69 | 50.31 | 52.12 |
| crossner_literature | 51.52 | 54.56 | 55.06 | 56.91 |
| crossner_music | 65.80 | 68.96 | 63.06 | 64.27 |
| crossner_politics | 55.26 | 56.41 | 62.47 | 66.52 |
| crossner_science | 56.08 | 60.85 | 58.31 | 55.47 |
| few_nerd | 52.37 | 55.14 | 51.49 | 47.22 |
| german_ler | 21.16 | 11.16 | 22.36 | 6.88 |
| hipe2020 | 45.46 | 39.56 | 41.22 | 29.45 |
| mobie | 30.65 | 24.44 | 32.47 | 29.66 |
| ronec | 40.13 | 37.01 | 38.86 | 31.55 |

+ Multilingual rows (multilingual_sentiment, xnli, german_ler, hipe2020, mobie, ronec): 2.5 Multi beats 2.5 Base on all six.
+ On English CrossNER, 2.5 Base (average 58.30) beats 2.5 Multi (54.85) by 3.45 points. GLiNER2 Base averages 59.06. This is larger than the 1 to 2 point gap our design expected (design section on the student).
+ Relation, JointIE and latency results for 2.5: not reported.

## 7. Pitfalls

Items marked "tested" were run in the API study or our tests. The others come from reading the code.

1. **Default LoRA targets leave the boundary head frozen** (tested). Use `lora_target_modules=["encoder", "all_task_heads"]` or name the heads. See 4.3.
2. **`_classification_loss` skips groups silently.** It skips a group with empty labels, labels shaped like a structure, or a label count that does not match the logit count (P/models/boundary/model.py:1617-1628). The docstring says a shape mismatch "raises" (lines 1599-1601). It does not. Inference raises on the same mismatch (P/classification/scoring.py:205-209). A broken label then trains nothing, with no warning.
3. **Surface-string targets mark every occurrence** (tested). See 3.5. Relation gold is the cross product of all head and tail copies.
4. **Validation changes the data.** With `validate_data=True` (default), the trainer sanitizes each record (P/training/data.py:316-357, 772-915):
   + It drops a whole entity type when one mention is not a substring of the text.
   + It drops a record with no task left. `"entities": {}` counts as no task; `{"person": []}` counts as a task.
   + It rebuilds `json_structures` through `Structure`, which adds `record_metadata` with `mode="natural"` and the first field as anchor (P/training/data.py:980-988, 927-940). Raw eval data is not validated, so train and eval can use different structure modes.
   + The check is a case-insensitive substring. The target search is a word match. "Paris" passes the check for "Parisian" but then raises "entity was not found" in training.
5. **`extract_json` uses record mode** (tested). When the anchor slot is absent, the output is `{}`. A `required_one` field never takes "absent" while a candidate exists (P/processing/records.py:36-55; P/models/boundary/records.py:760-765). Use `mode=None` or an entity schema for slots.
6. **`revision=` pins only `config.json`** (tested). `AutoExtractor.from_pretrained` passes Hub options only to the config load (P/auto.py:101-131). Load from a `snapshot_download(repo, revision=sha)` folder.
7. **Classification augmentation adds a wrong negative.** When labels are renamed to `label 1…` (p 0.5) and the true label is added back (p 0.5), the code adds the real name (P/processor.py:1071-1078). The true label maps to its synthetic name (line 1093), so the real name becomes a negative choice. This affects about 25% of classification groups in training.
8. **Renamed entity types get no description** unless you give `entity_descriptions`. The rename keeps only existing descriptions (P/processor.py:958). Classification and structure renames fall back to the real name (lines 902, 1054).
9. **Multi-label `classify_text` never returns an empty set.** It returns the argmax label when nothing passes the threshold (P/inference/runtime.py:416-420). Use `Classifier.batch_score` probabilities.
10. **README code has errors:**
    + `lora_targets=` is not a `TrainingConfig` field; the field is `lora_target_modules` (P/training/trainer.py:264-270). It raises `TypeError`.
    + `TrainingDataset.validate(strict=True, …)` has no `strict` parameter (P/training/data.py:1058).
    + `trainer.train(val_data=…)` has no such parameter; it is `eval_data` (P/training/trainer.py:1553-1557).
    + `model.load_adapter` exists only on the span model (P/models/span/model.py:844).
    + `train_gliner2()` loads with `GLiNER2.from_pretrained`, the span loader (P/training/trainer.py:2290-2292).
    + Tutorial 8 lists `{"person": []}` as valid; Tutorial 9 calls it an error. The code accepts it.
11. **Step-based schedules assume long runs.** Consistency warmup is 2,000 steps and soft IoU anneals over 20,000 steps, both in absolute global steps (P/training/trainer.py:1141-1158). A 300-step fine-tune keeps the consistency weight at 15% or less and keeps soft IoU almost fully on. Gold injection follows planned steps, so it scales.
12. **The last partial batch is dropped** each epoch when the dataset is larger than the batch (P/training/trainer.py:1529). With small sets this loses up to `batch_size - 1` examples per epoch.
13. **Abstention hides whole labels.** A label with null probability above 0.5 outputs nothing, whatever its span scores (P/models/boundary/engine.py:268-277). Uncertainty from emitted spans alone misses this.
14. **Relation decode merges by text** (P/models/boundary/engine.py:947-968). Span-level relation scoring needs care.
15. **Non-finite losses:** each loss term with NaN or inf becomes 0 (P/models/boundary/model.py:99-101). Under `strict_training=True`, a non-finite total raises at the next log step (P/training/trainer.py:1242-1252).
16. **Config versions:** the checkpoint says `transformers_version "5.8.0"`; the repo `pyproject.toml` pins `transformers<5` (web agent read of `pyproject.toml`). Our environment loads the model; pin versions in run logs.

## 8. What it means for us

We fine-tune `gliner2.5-multi-v1` with LoRA on 100 to 400 labelled sentences per task. The tasks are NER, multi-label classification, slot JSON and relations over given mentions.

+ **LoRA targets** - always include the heads we train. NER and slots need `boundary_head`; classification and relations-as-classification need `classifier`. Log the resolved target list and the trainable parameter count per run.
+ **Learning rate** - with LoRA only `task_lr` matters. It drives encoder adapters and heads at one rate.
+ **Targets from offsets** - keep our offset-based targets. Never feed plain strings to the stock processor for NER or slots.
+ **Negatives** - write empty-label sentences as `{"type": []}` for every type, or `true_label: []`. Never `"entities": {}`, which validation drops. This matters for research rule 4 (no gold filtering of the pool).
+ **Validation** - decide `validate_data` on purpose. If our loader already checks substrings, turn it off so the trainer does not change structure modes or drop types.
+ **Augmentation** - the default `SamplingConfig` renames labels and drops labels at high rates. With 100 examples this is a large share of the signal, and pitfall 7 adds wrong negatives. Consider a custom `SamplingConfig`, and fix it in the recipe for all arms.
+ **Classification loss** - add a test that every training group reaches the loss: label count equals logit count. The silent skip in pitfall 2 would hide a broken schema.
+ **Multi-label classification** - score with `Classifier.batch_score` and sigmoid, not `classify_text`, so an empty label set is possible.
+ **Slot JSON** - never use `extract_json` or record mode. Use an entity schema or `mode=None`. The empty-output confidence rule needs candidate scores, and abstention (pitfall 13) can hide a slot.
+ **Relations** - no public API scores given mentions. Our pair-classification design stays. The head and tail marker format must be the same in training and scoring.
+ **Uncertainty** - relation head and tail carry one pair score, so min(head, tail) equals that score.
+ **Pinning** - load from a local snapshot folder at `12fc40399dae672ce840c5e3c50a92340bff3c8c`. Log `gliner2==2.0.0`, torch and transformers versions.
+ **Short runs** - small budgets give few steps. Record steps per run; the consistency and soft IoU schedules (pitfall 11) behave differently from long pretraining.
+ **English gap** - the blog shows 2.5 Base above 2.5 Multi by 3.45 CrossNER points zero-shot. The reference ladder should measure this gap after LoRA.
+ **Paper claims** - cite the GLiNER2 paper for the span model only. For GLiNER2.5, cite the blog and model cards, and say it has no peer-reviewed report.

## Changelog
- 2026-09-25 23:25 CEST - Created from the paper, repo, model cards and installed code.
