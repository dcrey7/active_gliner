---
title: GLiNER2 2.0.0 API study - loading, schemas, confidence, training, and what it means for Active GLiNER
date: 2026-09-25 12:02 CEST
author: Claude (subagent)
type: research
status: frozen
---

# GLiNER2 2.0.0 API study

Question: what does the installed `gliner2` 2.0.0 package (GLiNER2 and GLiNER2.5) give us for the 4 tasks, and what must we still build?

Method:

1. I read the package source in the worktree venv. I read the dist-info METADATA (the PyPI README).
2. A helper agent read the GitHub repo, the tutorials, the releases, the open issues, the HF model cards, and the paper.
3. I ran the real models `fastino/gliner2.5-multi-v1` (the student) and `fastino/gliner2.5-base-v1` (comparison) on the RTX 3090, with English and French examples. I ran a tiny LoRA training run on each. Section 7 has the exact outputs.

Words used here:

- **Boundary model** - the GLiNER2.5 design. It finds span starts and ends, then scores start-end pairs. Class `BoundaryExtractor`.
- **Span model** - the older GLiNER2 design. It scores every span up to 8 words. Class `SpanExtractor` / `GLiNER2`.
- **Candidate** - a span that the boundary model proposes and scores. Only candidates can get a score.
- **Abstention** - a "no entity of this type here" score per label. If it is above 0.5, the label outputs nothing.
- **P** - the package root: `.venv/lib/python3.11/site-packages/gliner2/`. All file:line references are under P.

## 0. Short answer

1. Load with `AutoExtractor.from_pretrained(<repo or local folder>, map_location="cuda")`. It returns a `BoundaryExtractor` (P/auto.py:90-130). **`revision=` pins only `config.json`.** AutoExtractor does not pass `revision` on to the weight and tokenizer loads (P/auto.py:100-130). Pin by loading from a local snapshot folder (section 1.2). The student is now `fastino/gliner2.5-multi-v1` at `12fc40399dae672ce840c5e3c50a92340bff3c8c`; I ran both multi and base (section 7).
2. `include_confidence=True` gives one sigmoid score per output value, for every task. `include_spans=True` gives character offsets `start`, `end` (P/inference/runtime.py:989-996).
3. **Relations have one score per pair.** The boundary model puts the same pair score on head and tail (P/models/boundary/engine.py:886-893). So `min(head, tail)` equals the pair score.
4. **Classification: we can get all 10 label scores.** Use `Classifier.classify(...).to_dict()["probabilities"]` or `classify_text` with `cls_threshold=0.0`. Plain `classify_text` forces 1 label when no label passes the threshold (P/inference/runtime.py:416-420).
5. **Below-threshold scores are partly hidden.** The abstention gate drops whole labels (P/models/boundary/engine.py:268-277). Relations only see arguments above 0.2 (HF config `relation_argument_proposal_threshold`). We must read candidates through `JointIEEngine.score` or internal calls.
6. **`extract_json` is unsafe for MASSIVE.** On GLiNER2.5 it uses record mode with the first field as anchor. When the anchor slot is absent, the whole output is `{}` (section 7, "play the latest album by taylor swift on spotify").
7. The package trainer covers LoRA, fixed steps, dev evaluation, a custom dev metric, early stopping, best checkpoint, seed, bf16, and gradient accumulation. **But the default LoRA targets leave the boundary head frozen.** Use `lora_target_modules=["encoder", "all_task_heads"]`.
8. Code and weights are Apache-2.0.

## 1. Loading

### 1.1 Classes and model ids

| Class | What it loads | Source |
|---|---|---|
| `AutoExtractor.from_pretrained(path, *model_args, architecture=None, config=None, allow_architecture_override=False, **kwargs)` | Reads `config.json`, picks span or boundary class | P/auto.py:90-130 |
| `BoundaryExtractor` (GLiNER2.5) | Boundary checkpoints | P/models/boundary/engine.py:48-57 |
| `SpanExtractor`, `GLiNER2` | Span checkpoints only. `GLiNER2.from_pretrained` refuses boundary checkpoints | P/inference/engine.py:35-57, METADATA README line 84 |
| `Classifier.from_pretrained(repo, *, device=None, dtype=None, **load_kwargs)` | Wraps any extractor for constrained classification | P/classification/engine.py:81-93 |
| `JointIEEngine.from_pretrained(...)` (alias `JointIE`) | Wraps an extractor for typed entity-relation graphs | P/joint_ie/engine.py:154, P/joint_ie/__init__.py |

Model ids on the Hub (owner `fastino`, all Apache-2.0):

| Model id | Architecture | Encoder | Params | Source |
|---|---|---|---|---|
| `fastino/gliner2.5-small-v1` | boundary | deberta-v3-xsmall | 74M | README table |
| `fastino/gliner2.5-base-v1` | boundary | deberta-v3-base | 194M (193,581,591 F32 counted) | README table, https://huggingface.co/api/models/fastino/gliner2.5-base-v1 |
| `fastino/gliner2.5-multi-v1` (**our student**) | boundary | mdeberta-v3-base | 287M (287,355,159 counted; encoder 277,536,768, mostly the 250k-token embedding table) | README table, section 7 |
| `fastino/gliner2-base-v1`, `-large-v1`, `-multi-v1` | span | deberta-v3 base/large, mdeberta | 205M / 340M / ~205M | README table |
| `fastino/GLiNER2.5-Decide` | **span** (not boundary) | deberta-v3-large | 486M in safetensors | https://huggingface.co/fastino/GLiNER2.5-Decide/resolve/main/config.json |
| `fastino/GLiNER2.5-Decide-1B` | span | ettin-enc-from-dec-1b | ~1B | helper agent, model card |
| `fastino/GLiNER2.5-multi-Decide` | boundary | mdeberta-v3-base | 287M | helper agent, model card |

### 1.2 Load options

`from_pretrained` accepts only these keyword options (P/models/loading.py:22-41). Any other key raises `TypeError` (P/models/loading.py:45-59).

| Option | Kind | Effect |
|---|---|---|
| `revision`, `cache_dir`, `force_download`, `local_files_only`, `token`, `subfolder`, `proxies` | Hub | Passed to `hf_hub_download` |
| `map_location` | model | `model.to(map_location)`, for example `"cuda"` (P/models/loading.py:182-183) |
| `quantize` | model | `True` casts to fp16. `model.quantize("bf16")` also works (P/models/boundary/model.py:1203-1224) |
| `compile` | model | `torch.compile` on encoder and boundary parts (P/models/boundary/model.py:1226-1250) |
| `use_flashdeberta` | model | Optional FlashDeBERTa kernel (P/models/base.py:164-187) |
| `word_splitter` | model | `"whitespace"` (default) or `"char"` or a callable (P/models/base.py:226-246) |

Default dtype is fp32. The encoder is always cast to fp32 at build time (P/models/base.py:197-203).

**Revision pinning does not work through `AutoExtractor`.** This is a real bug, found by accident and confirmed from the source:

1. `AutoExtractor.from_pretrained` splits the options into model options and Hub options. It uses the Hub options (with `revision`) only to read `config.json`. It then calls `model_class.from_pretrained(path, config=config, **model_kwargs)` **without the Hub options** (P/auto.py:100-130).
2. So `encoder_config/config.json`, `model.safetensors`, and the tokenizer load from `main`.
3. Evidence: with `HF_HUB_OFFLINE=1`, `revision="1a8bc24e…"`, and the cache `refs/main` moved to `b0c10b23…`, the load failed with `OfflineModeIsEnabled: Cannot reach https://huggingface.co/fastino/gliner2.5-base-v1/resolve/main/encoder_config/config.json`.
4. `BoundaryExtractor.from_pretrained(repo, revision=sha)` does pass the Hub options to the config, encoder config, and weights (P/models/boundary/model.py:2076-2110). It still loads the tokenizer with no revision (P/models/boundary/model.py:2096-2101, P/models/base.py:44).
5. **Safe way:** `huggingface_hub.snapshot_download(repo, revision=sha)`, then `AutoExtractor.from_pretrained(<local folder>)`. All files then come from that folder. I used this for the final base run.

Revision facts (HF API, 25 Sep 2026):

| Model | Commit used | `main` today | Weight file | Note |
|---|---|---|---|---|
| `fastino/gliner2.5-multi-v1` (student) | `12fc40399dae672ce840c5e3c50a92340bff3c8c` | same (24 Sep 2026 22:08) | LFS `c1ff4ec0bc00…`, 1,149,461,028 bytes | Same weight and tokenizer hash at the older cached commit `a221b77a8baf…`. Loaded via AutoExtractor while `main` = `12fc4039…`, so the load matched the sha |
| `fastino/gliner2.5-base-v1` (comparison) | `1a8bc24e00dc7300b9017c81d63e3dcdabb26596` | `b0c10b23313ec3ff028821dff298dd743e010706` | LFS `7274094de2e0…`, 774,366,564 bytes | Same weight hash at `main`, `1a8bc24e…`, and the 21 Aug upload `cc885545e56e…`. Final run loaded from the local snapshot folder |

Recent commits on both repos change only the README and `SKILL.md`.

### 1.3 Batch inference

All public methods go through `batch_extract` (P/inference/runtime.py:69-152):

```python
batch_extract(texts, schemas, batch_size=8, threshold=0.5, num_workers=0,
              format_results=True, include_confidence=False, include_spans=False,
              max_len=None, overlap_policy=None) -> List[dict]
```

- `schemas` is one schema for all texts, or one per text (P/inference/runtime.py:92-97).
- Wrappers: `batch_extract_entities`, `batch_classify_text`, `batch_extract_json`, `batch_extract_relations` (P/inference/runtime.py:1329, 1383, 1476, 1592). Single-text forms call the batch form with batch size 1 (P/inference/runtime.py:1159-1168).
- Long text: `*_long` methods split into word chunks, `chunk_size=384`, `chunk_overlap=64`, and merge (P/inference/runtime.py:1170-1274).
- `max_len` cuts the text to N words. `None` means no cut at inference. The checkpoint config says `max_len: 4096`, and the trainer uses it (P/training/trainer.py:1513).
- `strict_extraction = True` makes one bad sample raise. Set it to `False` to get `{}` for that sample instead (P/inference/runtime.py:47-51, P/models/boundary/engine.py:185-193).
- The processor adds `"."` to any text that does not end in `. ! ?` (P/processor.py:521-525). Offsets into the original text stay correct.

## 2. Schema API

`Schema` is torch-free (P/inference/schema.py). `model.create_schema()` returns a new one (P/inference/runtime.py:61-63). Methods chain.

| Task | Call | Notes | Source |
|---|---|---|---|
| Entities | `.entities(entity_types, dtype="list", threshold=None, validators=None)` | `entity_types` is a str, a list, or a dict `{name: description}` or `{name: {"description", "threshold", "dtype", "validators"}}` | P/inference/schema.py:294-323, 409-424 |
| Relations | `.relations(relation_types, threshold=None)` | list or `{name: description}` or `{name: {"description", "threshold"}}`. Always 2 fields `head`, `tail`. **No entity-type constraint on head or tail** | P/inference/schema.py:426-461 |
| Classification | `.classification(task, labels, multi_label=False, cls_threshold=0.5, **kwargs)` | `labels` is a list or `{label: description}`. Extra keys (`class_act`, `prompt`, `examples`) pass through | P/inference/schema.py:267-292 |
| Structure (JSON) | `.structure(name, *, mode=None, anchor=None, occurrence_policy=None)` then `.field(name, dtype="list", choices=None, description=None, threshold=None, validators=None, cardinality=None, exclusive=False)` | `mode=None` = legacy aggregate decoder. `mode="natural"` = record decoder with an anchor field | P/inference/schema.py:246-265, 103-135 |
| Span attributes | `.entity_attributes({group: AttributeGroup(labels, multi_label=False, threshold=0.5, applies_to=None, qualify_labels=False)})` | Labels on extracted spans (for example sentiment) | P/inference/schema.py:59-76, 325-407 |
| Validators | `RegexValidator(pattern, mode="full", exclude=False, flags=re.IGNORECASE)` | Post-filter on span text | P/inference/schema.py:25-52 |
| From dict / JSON | `Schema.from_dict(...)`, `Schema.from_json(...)`, `.to_dict()` | Pydantic-validated | P/inference/schema.py:473-672 |

**Combined schemas.** One schema can hold entities, classifications, relations, and structures. `model.extract(text, schema)` runs all of them in one encoder pass. section 7 shows a real combined call.

**`extract_json` field strings.** `"name::str::description"`, `"name::list"`, `"name::[a|b|c]::str"` (P/inference/runtime.py:1663-1692). A field with no type is `list`. A choice list sets `str` unless you say otherwise.

**Gotcha: `extract_json` on GLiNER2.5 switches to record mode.** `_json_schema` sets `mode="natural"` when the model has a record head. The first field becomes the anchor. `str` fields become `required_one`, list fields `zero_or_more` (P/inference/runtime.py:1546-1579). If the anchor field has no match, the record is dropped. Use `Schema.structure(name)` with `mode=None` for the old one-instance behavior.

**Choices.** A `choices` field is scored at the choice word inside a prefix that is added before the text (P/processor.py:748-806, P/models/boundary/engine.py:656-760). It is not a span in the text.

**Descriptions.** Entity descriptions go into the prompt as `[DESCRIPTION] label: text` (P/processor.py:1111-1116). Relation descriptions become the relation prompt (P/processor.py:1026). Section 7 shows descriptions change scores and sometimes the label (base model, CoNLL "British lamb": person 0.66 without, miscellaneous 0.71 with; the multi model gives both labels at once).

**Scores depend on the schema.** The label names are part of the encoder input. The same span gets a different score when the label set changes (section 7: relation `role` for Marie Curie is 0.56 with 6 types and 0.63 with 4 types). Freeze the schema for all AL rounds.

## 3. Confidence

### 3.1 How scores are made (boundary model)

| Output | Score | Source |
|---|---|---|
| Entity span, JSON field span | `sigmoid(pair_logit / pair_temperature)`, temperature 1.0 | P/models/boundary/engine.py:84-87 |
| Relation pair | `sigmoid(relation_logit / relation_temperature)`, one number per (type, head, tail) | P/models/boundary/engine.py:826-834 |
| Classification, multi-label | `sigmoid(logit)` per label | P/inference/runtime.py:401-411 |
| Classification, single-label | `softmax(logits)` over labels | P/inference/runtime.py:406-411 |
| Choice field | `sigmoid` at the choice token in the prefix | P/models/boundary/engine.py:692-705 |
| Record field | `min(candidate probability, assignment probability)` | P/models/boundary/engine.py:1063-1080 |
| Abstention | `sigmoid(null_logit)` per entity label | P/models/boundary/engine.py:95-96 |

The paper describes span and classification scores the same way: dot product plus sigmoid, softmax for single-label (arXiv 2507.18546, appendix A, via helper agent).

### 3.2 `include_confidence` output format per task

With `include_confidence=True, include_spans=True` (real outputs in section 7):

| Task | Format |
|---|---|
| NER | `{"entities": {label: [{"text", "confidence", "start", "end"}, ...]}}`. Labels with no match give `[]` |
| Relations | `{"relation_extraction": {type: [{"head": {"text", "start", "end", "confidence"}, "tail": {... same "confidence"}}]}}`. Every requested type is present, empty or not |
| Classification, multi | `{task: [{"label", "confidence"}, ...]}`, only labels at or above `cls_threshold`, **or the top label if none pass** |
| Classification, single | `{task: {"label", "confidence"}}`, always the argmax, `cls_threshold` ignored |
| JSON, legacy (`mode=None`) | `{name: [ {field: [{"text", "confidence", "start", "end"}], scalar_field: {...} or None} ]}`, one instance |
| JSON, record mode | `{name: [record, record, ...]}`, or `{}` if no anchor |

`format_results=False` returns the unformatted dict, for example `{'entities': [OrderedDict([('person', [...])])]}` (section 7).

### 3.3 Can we get scores below the threshold?

| Task | `threshold=0.0` trick | Why | Better way |
|---|---|---|---|
| NER, labels present in text | Works. Gives all candidates of that label, after flat overlap removal within the label | P/models/boundary/engine.py:280-282 | same |
| NER, label absent | **Fails.** Abstention above 0.5 empties the label (section 7: `miscellaneous: 0` candidates at threshold 0.0) | P/models/boundary/engine.py:268-277 | Set `m.boundary_settings = dataclasses.replace(m.boundary_settings, abstention_threshold=1.0)` (tested, section 7), or use `JointIEEngine(m).score(...)` |
| Relations | **Fails.** Threshold 0.0 gave the same edges as 0.5 in both test sentences | Pairs come only from head and tail arguments with probability >= 0.2 (`relation_argument_proposal_threshold` in the HF config; P/models/boundary/relations.py:188-190) | `JointIEEngine.score` edges. **Do not** set `argument_threshold=0`: the dedup step then merges spans into whole-sentence spans (section 7) |
| Classification | Works with `cls_threshold=0.0` in the task dict (not the `threshold` argument) | P/inference/runtime.py:414-417 | `Classifier(m).batch_score(...)`, `ClassificationScores.probability(task, label)` for every label (P/classification/scoring.py:96-107) |
| JSON legacy | Works for fields that have candidates. Gives many junk spans | same as NER, no abstention on structure fields | `JointIEEngine.score` or entity schema with abstention off |
| JSON record mode | Not useful. Low threshold makes many fake records (section 7) | P/models/boundary/engine.py:1213-1245 | avoid record mode |

**Raw candidate access (tested).** `JointIEEngine(model).score(text, joint_schema)` returns a `CandidateScoreSet` (P/joint_ie/engine.py:213, P/joint_ie/scoring.py:284-412). It has:

- `mentions`: `MentionScore(query_id, entity_type, start, end, logit, probability, ...)` for every valid candidate of every label, no threshold, no abstention. Offsets are word-token indices, half-open. `text_tokens`, `start_mappings`, `end_mappings` map them to characters (P/joint_ie/candidate_scores.py:27-80).
- `edges`: `ScoredRelationEdge(relation_type, head, tail, logit, probability, ...)` for scored relation pairs (P/joint_ie/candidate_scores.py:56-66). The pairs still pass the 0.2 argument filter.

Probe: 3 entity types on a 10-word sentence gave 234 mentions and 2 edges (section 7).

**Lower-level path (tested, private API).** `ExtractorCollator(m.processor, is_training=False, architecture="boundary")`, then `m._encode_core(batch)`, then `m.boundary_head(..., return_candidates=True)`. The output has `candidates.pair_logits`, `candidates.indices`, `candidates.valid_mask`, `null_logits`, `count_log_rates` (P/models/boundary/engine.py:67-96). Probe: `null probs [[0.0026, 0.9972]]` for `person` and `disease`; `max prob per query [[0.9986, 0.0011]]`. These are private names and may change.

**Candidate budget.** The HF config caps candidates: shared pool of 192, at least 8 per query, `start_top_k`/`end_top_k` 24 (https://huggingface.co/fastino/gliner2.5-base-v1/resolve/main/config.json). A span outside the pool has no score at all.

**Per-label thresholds.** Entities: `threshold` per label in the dict form (P/models/boundary/engine.py:198-226). Relations: per type in `relations({...: {"threshold": x}})` (P/models/boundary/engine.py:849-853). Structure fields: `field(..., threshold=x)`. Classification: `cls_threshold` per task, not per label.

**Calibration.** Temperatures exist in config (`pair_temperature`, `relation_temperature`, `classification_temperature`, all 1.0; P/configuration.py:109-112). `ClassificationConfig.calibrator` exists (P/classification/engine.py:46) but the helper agent reports it is never read (open PR #173, https://github.com/fastino-ai/GLiNER2/pull/173). Unconfirmed by me.

## 4. Training

### 4.1 Classes

- `TrainingConfig` - a dataclass (P/training/trainer.py:89-349).
- `ExtractorTrainer(model, config, processor=None, train_data=None, eval_data=None, compute_metrics=None)` (P/training/trainer.py:683-761). `GLiNER2Trainer` is an alias (P/training/trainer.py:2235).
- `trainer.train(train_data=None, eval_data=None) -> dict` (P/training/trainer.py:1553-1919). The README says `val_data=`; the real name is `eval_data`.
- `train_gliner2(...)` loads with `GLiNER2.from_pretrained`, which is span-only (P/training/trainer.py:2290-2292). Do not use it for GLiNER2.5.

### 4.2 `TrainingConfig` fields and defaults (P/training/trainer.py:180-270)

| Group | Fields (default) |
|---|---|
| Run | `output_dir="./output"`, `experiment_name="gliner2"`, `seed=42`, `deterministic=False`, `debug=False` |
| Length | `num_epochs=10`, `max_steps=-1` (if > 0 it wins over epochs; P/training/trainer.py:1630-1636), `max_train_samples=-1`, `max_eval_samples=-1`, `max_len=None` (falls back to the model config, 4096) |
| Batches | `batch_size=2`, `eval_batch_size=8`, `gradient_accumulation_steps=1`, `num_workers=4`, `pin_memory=True`, `prefetch_factor=2`, `group_by_length=True`, `length_group_window_batches=50` |
| Optimizer | AdamW. `encoder_lr=1e-5`, `task_lr=5e-4`, `weight_decay=0.01`, `adam_beta1=0.9`, `adam_beta2=0.999`, `adam_epsilon=1e-8`, `max_grad_norm=1.0`, `fused_optimizer=True` |
| Schedule | `scheduler_type="linear"` (also `cosine`, `cosine_restarts`, `constant`), `warmup_ratio=0.1`, `warmup_steps=0`, `num_cycles=0.5` |
| Precision | `fp16=None`, `bf16=None`. If you set neither, fp16 becomes True (P/training/trainer.py:272-279), **but for a boundary model the trainer switches to bf16** (P/training/trainer.py:694-705). Tested: `True False` before, `False True` after trainer init |
| Eval and save | `eval_strategy="steps"` (`"epoch"`, `"no"`), `eval_steps=500`, `save_total_limit=3`, `save_best=True`, `metric_for_best="eval_loss"`, `greater_is_better=False` |
| Early stop | `early_stopping=False`, `early_stopping_patience=3`, `early_stopping_threshold=0.0` |
| Logging | `logging_steps=1`, `logging_first_step=True`, `report_to_wandb=False`, `wandb_project`, `wandb_entity`, `wandb_run_name`, `wandb_tags`, `wandb_notes` |
| LoRA | `use_lora=False`, `lora_r=16`, `lora_alpha=32.0`, `lora_dropout=0.0`, `lora_use_dora=False`, `lora_target_modules=["encoder", "span_rep", "classifier", "count_embed", "count_pred"]`, `save_adapter_only=True` |
| Robustness | `strict_training=True`, `ignore_nonfinite_losses=False`, `skip_step_errors=False`, `allow_invalid_samples=False`, `on_capacity_exceeded="raise"` |
| Boundary only | `gold_injection_start=1.0`, `gold_injection_end=0.25`, `gold_injection_hold_frac=0.15`, `log_proposal_metrics=True`, `dry_run_recall_steps=0`, `gate_recall=0.97`, `gate_long_recall=0.93` |
| Speed | `compile_model=False`, `gradient_checkpointing=False`, `allow_tf32=True`, `float32_matmul_precision="high"` |
| DDP | `local_rank=-1`, `ddp_consensus_check`, `ddp_find_unused_parameters`, `ddp_static_graph` |

### 4.3 LoRA

- The trainer freezes every weight, then calls `model.apply_lora(r, alpha, dropout, targets, use_dora)` (P/training/trainer.py:849-871). This uses PEFT `get_peft_model` (P/models/boundary/model.py:1252-1280). `trainer.model` becomes a `PeftModel`.
- **LoRA touches only `nn.Linear` layers** (P/training/lora_targets.py:66-89). LayerNorms and other head weights stay frozen.
- **With LoRA, all trainable weights use `task_lr`. `encoder_lr` is ignored** (P/training/trainer.py:1352-1360).
- `apply_lora` changes the model object in place. The `model` you passed now has LoRA layers too. Reload the base model for each AL round.

Target names that exist in `gliner2.5-base-v1` and `gliner2.5-multi-v1` (tested with `_resolve_targets`; the counts and names are the same for both, because mDeBERTa-v3-base has the same layer layout; example module `encoder.encoder.layer.0.attention.output.dense`). The word-embedding table is not a Linear layer, so LoRA never touches it:

| `lora_target_modules` | Linear layers | Top-level modules |
|---|---|---|
| `["encoder"]` | 72 | encoder (`query_proj`, `key_proj`, `value_proj`, `dense`) |
| default `["encoder","span_rep","classifier","count_embed","count_pred"]` | 74 | encoder, classifier. **boundary_head, relation_scorer, record_decoder stay frozen** |
| `["encoder", "all_task_heads"]` | 131 | encoder, classifier, boundary_head, record_decoder, relation_scorer |
| `["all_task_heads"]` | 59 | the 4 heads |
| `["extractive_head"]` | 43 | boundary_head |
| `["relation_head"]` | 6 | relation_scorer |

Other names: `"encoder.query"`, `"encoder.key"`, `"encoder.value"`, `"encoder.dense"` (substring match), `"classification_head"`, `"record_head"`, `"classifier"`, `"boundary_head"`, `"relation_scorer"`, `"record_decoder"` (P/training/lora_targets.py:34-64). Head sizes: classifier 1.18M, boundary_head 1.62M, record_decoder 1.11M, relation_scorer 5.91M parameters.

**Recommended settings from Fastino** (tutorial 10, https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/10-lora_adapters.md, via helper agent):

| Data size | `lora_r` | `lora_alpha` | Epochs |
|---|---|---|---|
| under 1k | 4 | 8 | 10 |
| 1k-10k | 8 | 16 | 5 |
| over 10k | 16 | 32 | 3 |

The tutorial example uses batch 8, gradient accumulation 2, `task_lr=5e-4`, dropout 0.0, targets `["encoder"]`. The README LoRA example uses `lora_targets=[...]`, which is **not a field** and raises `TypeError` (METADATA README lines 1287-1299 vs P/training/trainer.py:264-270). The paper used 5 epochs, backbone LR 1e-5, task LR 2e-5, 1,000 warmup steps (helper agent, arXiv 2507.18546).

### 4.4 Save, load, merge, several adapters

- With `use_lora=True` and `save_adapter_only=True`, each checkpoint is PEFT-native: `adapter_config.json`, `adapter_model.safetensors`, `README.md` (P/training/trainer.py:2101-2119; tested).
- Folders: `checkpoint-<step>` (every `eval_steps`, even with no eval data; P/training/trainer.py:1842-1852), `best` (when dev metric improves; P/training/trainer.py:2017-2021), `final` (P/training/trainer.py:1899-1900), `training_config.json`. Old `checkpoint-*` folders beyond `save_total_limit` are deleted; `best` and `final` stay (P/training/trainer.py:2191-2205).
- Load (tested): `PeftModel.from_pretrained(AutoExtractor.from_pretrained(base, revision=sha), "out/best")`. The `PeftModel` forwards `extract_entities` and others. `merge_and_unload()` returns a plain `BoundaryExtractor` with the same outputs (section 7).
- **`BoundaryExtractor` has no `load_adapter` / `save_adapter` / `merge_lora`.** Those exist only on the span model and are deprecated (P/models/span/model.py:844-899). The README `model.load_adapter(...)` example works only for span models.
- Several adapters: use PEFT directly (`load_adapter(path, adapter_name=...)`, `set_adapter(...)`). The package has no own helper for boundary models. Tutorial 11 covers adapter switching for the API; I did not test it on boundary models (unconfirmed).
- After early stopping, `trainer.model` holds the **last** weights, not the best. Reload `best` (P/training/trainer.py:1848-1851, 2017-2021).
- `trainer.load_checkpoint(path)` exists (P/training/trainer.py:2207-2231).

### 4.5 Evaluation, early stopping, seeds

- Built-in evaluation is **loss only**: `eval_loss`, `eval_classification_loss`, `eval_structure_loss`, `eval_count_loss`, plus proposal-recall ratios (P/training/trainer.py:1995-2003).
- `compute_metrics(model, eval_dataset) -> dict` is merged into the eval metrics (P/training/trainer.py:2005-2006). Set `metric_for_best` to one of its keys and `greater_is_better=True`. Tested: the hook ran at steps 4, 8, 12, and `best_metric` followed it.
- Early stopping counts evaluations with no gain of more than `early_stopping_threshold` (P/training/trainer.py:2025-2038).
- `_setup_seed` seeds `random`, `numpy`, `torch`, `torch.cuda`. `deterministic=True` sets cuDNN deterministic; otherwise `cudnn.benchmark=True` (P/training/trainer.py:777-787). The train dataset shuffle and the length-grouped sampler use `config.seed` (P/training/trainer.py:1334-1340, 1500-1507). Full GPU determinism is not promised (unconfirmed).
- Mixed precision uses `torch.amp.autocast`; `GradScaler` only for fp16 (P/training/trainer.py:1644-1649).
- Gradient accumulation handles a partial last window (P/training/trainer.py:1862-1872).

### 4.6 Training data format

JSONL line: `{"input": text, "output": {...}}`, or `InputExample` objects, or `TrainingDataset`, or a list of dicts (P/training/trainer.py:7-13, P/training/data.py:917-943).

| Task | `output` keys | Python |
|---|---|---|
| NER | `"entities": {label: [surface strings]}`, optional `"entity_descriptions"`. An empty list is a negative label | `InputExample(text, entities={...}, entity_descriptions={...})` |
| Relations | `"relations": [{type: {"head": str, "tail": str}}]` | `Relation(name, head=..., tail=...)` (P/training/data.py:604-669) |
| Classification | `"classifications": [{"task", "labels", "true_label": [..], "multi_label", "label_descriptions", "prompt", "examples"}]` | `Classification(...)`. Several true labels set `multi_label=True` (P/training/data.py:377-450) |
| JSON | `"json_structures": [{name: {field: str or [str] or {"value", "choices"}}}]`, `"json_descriptions"`, `"record_metadata"` | `Structure(name, mode="natural", anchor=None, **fields)`. **Default mode is `"natural"` (record mode)**; pass `mode=None` for legacy (P/training/data.py:502-546) |

Important details:

1. **Labels are surface strings, not offsets.** The processor finds them by word-token match and marks **every occurrence** (P/processor.py:1206-1237). Tokens are lower-cased (P/processor.py:571), so the match ignores case. A string that is an entity in one place and not in another is labeled in both places.
2. An example needs at least one task. `entities={}` alone is invalid; `entities={"person": []}` is a valid negative (P/training/data.py:762-764).
3. Validation checks that every string is a substring of the text (P/training/data.py:716-766). A substring inside a longer word (for example `Lithium` inside `Lithium-induced`) passes this check but cannot match a word token (see 5.4).
4. **Training uses random augmentation by default** (`SamplingConfig`, P/processor.py:242-262; only in training, P/processor.py:814):
   - 20% of entity queries get synthetic names `entity 1..n`.
   - Relations: each instance is dropped with p = 0.2; head and tail order is swapped with p = 0.2.
   - Classification: labels shuffled, up to 50% of labels dropped, synthetic names with p = 0.5.
   - JSON: 20% of structures and 20% of fields dropped.
   - Change it with `model.processor.sampling_config = SamplingConfig(...)` (unconfirmed that the trainer keeps this object; it reads `model.processor`, P/training/trainer.py:710).

## 5. Other useful facts

### 5.1 Hosted API

- `GLiNER2API(api_key=None, api_base_url=None, timeout, max_retries)`; alias `API`. Key from `PIONEER_API_KEY`. Base URL `https://api.fastino.ai`, or `GLINER2_API_BASE_URL` (P/api_client.py:296-329).
- Same method names as the local model, plus local batching and long-text chunking (METADATA README lines 126-143).
- Open issue #34: schema thresholds are dropped in the API client `extract()` (https://github.com/fastino-ai/GLiNER2/issues/34, via helper agent).
- The Fastino platform (training jobs, auto-labeling) is covered in `docs/research/2026-09-25-1206-fastino-platform-scan.md`.

### 5.2 GLiNER2.5-Decide

- A classification specialist: intent, routing, sentiment, severity, moderation, yes/no questions over a passage (https://huggingface.co/fastino/GLiNER2.5-Decide, via helper agent).
- The English Decide models are **span** checkpoints despite the "2.5" name (config above).
- Use: `AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide").classify_text(text, {task: labels})`.
- It could be a stronger zero-shot classifier baseline for Hallmarks. It is a different student architecture, so it does not fit our GLiNER2.5-base LoRA student. Not tested.

### 5.3 Other features

- `Classifier` + `ClassificationSchema`: `.single`, `.multi`, `.ordinal`, `.task(name, labels, *, min_labels=0, max_labels=None, threshold=0.5, candidate_threshold=None, activation="auto", temperature=1.0, default=None, instruction=None, examples=())`, `.constrain(...)` (P/classification/schema.py:215-246). Decoders `auto|independent|exact|beam` (P/classification/engine.py:36-64).
- `JointIEEngine` + `JointSchema`: `.entities([...])`, `.relation(name, head_types, tail_types, ...)` with typed endpoints and graph constraints. `JointIEConfig(candidate_threshold=0.05, relation_role_threshold=0.05, top_k_entities=32, ...)` (P/joint_ie/engine.py:18-49).
- `overlap_policy`: `allow`, `nested`, `flat`/`disallow`, `longest` (P/inference/overlap.py:17-28). The boundary default `flat` applies **within one label only** (P/models/boundary/engine.py:280-282). Across labels, spans can overlap (section 7: "German" is location 0.72 and person 0.22 at threshold 0.01).

### 5.4 Gotchas found

1. **Word splitting keeps hyphen words whole.** The regex `\w+(?:[-_]\w+)*` makes `Lithium-induced` one token (P/processing/word_splitter.py:25-32). The model cannot output `Lithium` alone (section 7 BC5CDR). BC5CDR has many `X-induced` chemicals. A custom splitter can fix this: `model.set_word_splitter(callable)` (P/models/base.py:226-246). It must be the same at training and inference.
2. **Environment:** with transformers 4.57.6 the tokenizer load fails with `ImportError: requires the protobuf library`. The package fallback does not catch it (P/models/base.py:44-56). I ran with `uv run --no-sync --with protobuf --with sentencepiece`. Add `protobuf` to our dependencies (sentencepiece may not be needed; unconfirmed).
3. **Forced label in multi-label `classify_text`:** when no label passes `cls_threshold`, it returns the top label anyway (P/inference/runtime.py:416-420). section 7: an off-topic sentence got `resisting cell death` at 0.037.
4. **Single-label ignores `cls_threshold`** (P/inference/runtime.py:422-424; open issue #66).
5. The `docs/boundary_architecture.md` files linked from the README return 404 (helper agent).
6. Open PR #176 says relation and record confidences are "at the wrong level" and proposes separate head, tail, and relation scores (https://github.com/fastino-ai/GLiNER2/pull/176, via helper agent). A future release may change the relation output format. Pin the package version.
7. Open PR #160: `unload_adapter()` does nothing on PEFT loads (via helper agent).

### 5.5 Licence

- Code: Apache-2.0 (METADATA line 6).
- Weights: `license: apache-2.0` on `gliner2.5-base-v1` (HF API tags) and on the Decide cards (helper agent).

## 6. Implications for our design

### 6.1 What the package already covers (delete our own code)

| Our planned piece | Package piece | Action |
|---|---|---|
| Model loading, device, dtype, revision | `AutoExtractor.from_pretrained(<local snapshot folder>, map_location=, quantize=)` | Use it. Get the folder with `snapshot_download(repo, revision=sha)`. Do not rely on `revision=` (section 1.2) |
| Batched inference with offsets | `batch_extract(..., include_confidence=True, include_spans=True)` | Use it |
| Per-entity and per-slot-value confidence | `confidence` per value | Use it |
| All 10 classification label scores | `Classifier(m).batch_score(texts, schema)` then `.probability(task, label)`; or `cls_threshold=0.0` | Use it. Do not write our own head call |
| Relation pair score | one `confidence` per pair (same on head and tail) | Use it |
| LoRA training loop | `ExtractorTrainer` + `TrainingConfig(use_lora=True, max_steps=..., eval_strategy="steps", eval_steps=..., compute_metrics=..., metric_for_best=..., greater_is_better=True, early_stopping=True, seed=...)` | Use it. Write only `compute_metrics` (dev F1) |
| Checkpoints, best model | PEFT adapter folders `best`, `final` | Use them. Load with `PeftModel.from_pretrained` |
| Training-data format and validation | `InputExample`, `TrainingDataset.validate()` | Use them |
| Long documents | `*_long` methods | Use if a text is over the limit (not needed for our sentence-level sets) |

### 6.2 Gaps we must build

1. **Below-threshold candidate scores (NER, slots).** Needed for "no prediction" cases. Build one small adapter over `JointIEEngine.score` (public, tested) that returns, per sentence and label, the max candidate probability. Fallback: turn off abstention (`abstention_threshold=1.0`) and call with `threshold=0.0`. Record which path we use. Both change nothing in the model weights.
2. **Relation empty-output score.** Pairs exist only if both arguments reach 0.2. So for most sentences with no relation, there is no candidate at all. Use `JointIEEngine.score(...).edges`; if there is no edge, max candidate probability = 0 and c = 1. Say in the paper that relation "absence confidence" is gated by the argument filter.
3. **Cross-label flat NER decoding.** The package allows one span to carry two labels. CoNLL, BC5CDR, MIT Movie are flat. Add a decoder that keeps the highest-scoring label per overlapping group, or report the rate of cross-label overlaps. Freeze the rule before runs.
4. **Hyphen splitter for BC5CDR.** Measure how many gold spans are not word-token aligned under the default splitter. If the rate is not small, use a custom splitter at training and inference.
5. **Hallmarks empty label set.** Use `Classifier` (multi task has `min_labels=0`, so the output can be empty; tested) or our own threshold on the 10 probabilities. Never use plain `classify_text` for the decision.
6. **MASSIVE slots.** Do not use `extract_json` (record mode, drops records with no anchor). Two safe options, both tested:
   - Entity schema with one label per slot type. Highest scores, simplest, same code as NER.
   - `Schema.structure("slots")` with `mode=None` and list fields. Needs `Structure(..., mode=None)` in training data.
   I recommend the entity schema; the output maps directly to `{slot_type: [values]}`.
7. **LoRA targets.** Set `lora_target_modules=["encoder", "all_task_heads"]` (or at least `["encoder", "extractive_head", "relation_head"]` for relations). The default leaves the boundary head frozen.
8. **Fresh base per round.** LoRA injection mutates the model object. Load a new base for each AL round and seed.
9. **Augmentation choice.** Decide whether to keep `SamplingConfig` defaults. Relation instance dropping (p = 0.2) turns gold relations into negatives for that step. Freeze the choice and record it.
10. **Environment pin.** Add `protobuf`; pin `gliner2==2.0.0`; load the model from a `snapshot_download` folder at the sha, because `AutoExtractor(revision=)` does not pin weights.
11. **Multilingual student.** `gliner2.5-multi-v1` gave the same output formats and correct French outputs (section 7). Its scores differ from base on the same English sentences (for example BC5CDR `Lithium-induced` chemical: base 0.75, multi below 0.5). Tune thresholds on dev for multi, not from base results.

### 6.3 Conflicts with our design assumptions

| Design rule (docs/research/2026-09-25-1113-full-merge-design.md) | Finding | Change |
|---|---|---|
| Relations: "if head and tail differ, use min(head, tail) and call it a proxy" (line 44) | In GLiNER2.5 head and tail always carry the same pair score (P/models/boundary/engine.py:886-893; section 7) | Drop the proxy wording. Relation confidence = the pair score. Keep a note that PR #176 may change this |
| Relations: "directed typed triples", entity types may count | `Schema.relations` has no argument types; the arguments are free spans | If entity types count, use `JointIEEngine` typed relations or a separate NER pass. Decide before runs |
| Classification: `min abs(p - t)` over all 10 labels (line 30, 45) | All 10 probabilities are available (tested). `classify_text` forces one label | Rule is fine. Compute from `Classifier` probabilities, not from `classify_text` output |
| Slot JSON empty-output rule: c = 1 - max below-threshold slot score (line 46) | Absent labels are hidden by abstention; JSON record mode gives `{}` | Rule is fine if we read candidates through `JointIEEngine.score` or abstention off. Define "candidate" = any proposed span of any slot label |
| NER: "no prediction = 0" (line 27) | Consistent. Candidate scores exist if we want a finer rule later | No change |
| Student = GLiNER2.5-base LoRA | Trainer defaults to bf16 for boundary; LoRA uses `task_lr` for all | Record `task_lr` as the only LoRA learning rate |

## 7. Hands-on outputs

### 7.1 Setup

| Item | Value |
|---|---|
| Package | `gliner2` 2.0.0, torch 2.14.0+cu130, transformers 4.57.6, peft 0.21.0 |
| Run command | `uv run --no-sync --with protobuf --with sentencepiece python <script>` from the worktree (protobuf is missing from the venv; see 5.4) |
| GPU | NVIDIA GeForce RTX 3090, fp32 inference, no `quantize`, no `compile` |
| Student | `fastino/gliner2.5-multi-v1`, commit `12fc40399dae672ce840c5e3c50a92340bff3c8c` |
| Comparison | `fastino/gliner2.5-base-v1`, commit `1a8bc24e00dc7300b9017c81d63e3dcdabb26596`, loaded from the local snapshot folder |
| Scripts | `/tmp/g2probe/handson2.py` (both models, same sentences), `probe2.py` (JointIE, Classifier, internals), `probe3.py` (runtime knobs), `train_smoke.py`, `train_smoke_multi.py`. Scripts are outside the repo and not kept |

Sentences: CoNLL-style news (English and one French), BC5CDR chemical/disease, MIT Movie query, CrossRE-style relations (English and French), 10 Hallmarks labels (English and French), MASSIVE-style commands (English and French), one combined schema.

### 7.2 Timing and GPU memory

Mean of 5 calls after 1 warm-up call. Batch numbers are total time divided by the number of sentences (80 NER sentences, 66 classification sentences).

| Measure | multi-v1 (student) | base-v1 |
|---|---|---|
| Load time (from cache) | 5.7 s | 1.9 s |
| GPU memory after load | 1,097 MiB | 740 MiB |
| Peak GPU memory, whole inference script | 1,546 MiB | 1,068 MiB |
| NER, 1 sentence, 4 labels | 9.23 ms | 9.40 ms |
| Relations, 1 sentence, 6 types | 11.97 ms | 11.91 ms |
| Classification, 1 sentence, 10 labels | 7.02 ms | 7.17 ms |
| JSON slots, 1 sentence, 8 fields | 11.94 ms | 11.88 ms |
| Combined schema, 1 sentence | 11.31 ms | 11.67 ms |
| NER batch 8, per sentence | 1.49 ms | 1.40 ms |
| NER batch 32, per sentence | 1.00 ms | 0.85 ms |
| Classification batch 32, per sentence | 1.97 ms | 1.51 ms |
| LoRA smoke training peak GPU memory (batch 4, bf16, r 8, `encoder`+`all_task_heads`) | 1,563 MiB | 1,173 MiB |

Single calls cost about 7-12 ms for both models. Batching gives about 10x more sentences per second. A 12k-sentence pool scores in well under a minute per task on the 3090.

### 7.3 Answers to the 5 checks

1. **NER.** Works in English and French. Offsets are correct. Descriptions change scores. At threshold 0.0, labels with no match still give 0 candidates (abstention). One span can carry 2 labels (multi: "British lamb" is person 0.57 and miscellaneous 0.66).
2. **Relations.** One score per (type, head, tail). Head and tail carry the same number. Thresholds 0.01 and 0.0 add no pairs.
3. **Classification, 10 labels, multi-label.** With `cls_threshold=0.0`, or with `Classifier`, every label gets a score. With `cls_threshold=0.5`, an off-topic sentence still gets 1 label (multi: `sustaining proliferative signaling` 0.18; base: `resisting cell death` 0.037). `Classifier` returns an empty list for it.
4. **JSON slots.** Each value has its own confidence and offsets. `extract_json` (record mode) returns `{}` for "play the latest album by taylor swift on spotify", "what is the weather like", and "olly tell me a joke" on both models, because the anchor field `time` is absent. `Schema.structure(mode=None)` and the entity schema return the slots. With no match, the entity schema returns `[]` per label; legacy structure returns `{name: [instance with all fields empty]}` or drops the instance when all fields are empty (P/models/boundary/engine.py:590-591).
5. **Combined schema.** One call returns `event`, `entities`, `topic`, and `relation_extraction` together (multi: topic `science` 0.89; base: topic `politics` 0.77).

### 7.4 Internal probes (base-v1, same code path as multi)

`probe2.py` output (exact):

```text
scoreset type CandidateScoreSet mentions 234 edges 2 roles 0
  edge works_for ('person', 0, 1) ('organization', 3, 6) 0.99301
  edge lives_in ('person', 0, 1) ('location', 9, 11) 0.9821
  n edges per type {'works_for': 1, 'lives_in': 1, 'founded': 0}
  mention person 0 1 0.99954
  mention location 9 11 0.9992
  mention organization 3 6 0.99856
  mention person 0 2 0.03259
  mention organization 0 6 0.02736
  mention location 8 11 0.02732
joint extract JointResult ['default_include_confidence', 'default_include_spans', 'entities', 'entities_by_type', 'entity', 'feasible', 'get_entity', 'incoming', 'neighbors', 'outgoing', 'relations', 'relations_by_type', 'relations_of', 'text', 'to_dict', 'to_networkx']
{'entities': [{'id': 'e1', 'type': 'person', 'text': 'John', 'start': 0, 'end': 4, 'confidence': 0.9995429629791984}, {'id': 'e2', 'type': 'organization', 'text': 'Apple Inc.', 'start': 15, 'end': 25, 'confidence': 0.9985573023768816}, {'id': 'e3', 'type': 'location', 'text': 'San Francisco', 'start': 39, 'end': 52, 'confidence': 0.9992026886504899}], 'relations': [{'type': 'lives_in', 'head': 'e1', 'tail': 'e3', 'confidence': 0.9820970561669689}, {'type': 'works_for', 'head': 'e1', 'tail': 'e2', 'confidence': 0.9930119957034678}]}
cls probs {'evading apoptosis': 0.00027, 'angiogenesis': 0.00069, 'invasion and metastasis': 0.00065, 'genomic instability': 0.00084}
cls result {'hallmarks': {'value': [], 'confidence': None, 'probabilities': {'evading apoptosis': 0.0002740904107271443, 'angiogenesis': 0.0006886382464383071, 'invasion and metastasis': 0.0006487870837459603, 'genomic instability': 0.000835387910273016}}, '_meta': {'feasible': True, 'decoder': 'independent', 'exact': True, 'objective': 0.0, 'violations': []}}
null probs [[0.0026345252990722656, 0.9971832633018494]]
candidates indices shape (1, 2, 192, 2) valid per query [[36, 36]]
max prob per query [[0.9985854625701904, 0.0011104795848950744]]
count_log_rates [[3.189588785171509, 0.005998918320983648]]
```

(Text for the JointIE lines: "John works for Apple Inc. and lives in San Francisco." Text for the internals: "Barack Obama met Angela Merkel in Berlin." with labels person, disease. `count_log_rates` printed after `exp`.)

`probe3.py` output (exact), runtime knobs:

```text
default argument_threshold 0.2
REL thr0 default: {'role': [('Marie Curie', 'University of Paris', 0.626)], 'origin': [('Marie Curie', 'Warsaw', 0.9934)], 'win-defeat': [], 'physical': []}
REL thr0 argument_threshold=0: {'role': [('Marie Curie was born in Warsaw and later worked at the University of Paris', 'was born in Warsaw and later worked at the University of Paris.', 0.6419), ('University of Paris.', 'was born in Warsaw and later worked at the University of Paris.', 0.0784)], 'win-defeat': [('Marie Curie was born in Warsaw', 'Marie Curie was born in Warsaw and later worked at the University of Paris', 0.6692), ('Marie Curie was born in Warsaw', 'University of Paris.', 0.1099), ('later', 'Marie Curie was born in Warsaw and later worked at the University of Paris', 0.0003), ('worked', 'Marie Curie was born in Warsaw and later worked at the University of Paris', 0.0008), ('University of Paris', 'Marie Curie was born in Warsaw and later worked at the University of Paris', 0.0527)], 'physical': [('Marie Curie was born in Warsaw', 'Marie Curie was born', 0.5497), ('Marie Curie was born in Warsaw', 'in Warsaw', 0.0213), ('Marie Curie was born in Warsaw', 'Warsaw and', 0.5906), ('Marie Curie was born in Warsaw', 'later worked', 0.0048), ('Marie Curie was born in Warsaw', 'worked at the University of Paris', 0.1583), ('Marie Curie was born in Warsaw', 'University of Paris.', 0.0681), ('later', 'worked at the University of Paris', 0.001), ('University of Paris', 'Marie Curie was born', 0.0247), ('University of Paris', 'Warsaw and', 0.1101), ('University of Paris', 'worked at the University of Paris', 0.0778)], 'origin': [('Marie Curie was born in Warsaw and later worked at the University of Paris', 'Marie Curie was born in Warsaw and later worked at the University of Paris', 0.994), ('Marie Curie was born in Warsaw and later worked at the University of Paris', 'Paris.', 0.0095), ('Paris.', 'Marie Curie was born in Warsaw and later worked at the University of Paris', 0.015)]}
NER thr0 default abstention: {'person': 6, 'disease': 0}
NER thr0 abstention off: {'person': [('Barack Obama', 0.99859), ('Angela Merkel', 0.99518), ('Berlin', 0.01469)], 'disease': [('Berlin', 0.00111), ('Angela Merkel', 0.00019), ('Barack Obama', 5e-05)]}
```

Reading: turning abstention off is safe and gives below-threshold scores for absent labels. Setting the relation argument threshold to 0 breaks the output (whole-sentence spans after dedup). Do not use it.

### 7.5 LoRA training smoke test

Setup: 6 examples x 4 (NER, negative NER, multi-label classification, relation, legacy structure) = 24 train, 6 dev. `TrainingConfig(max_steps=12, batch_size=4, eval_batch_size=4, eval_strategy="steps", eval_steps=4, use_lora=True, lora_r=8, lora_alpha=16.0, lora_target_modules=["encoder", "all_task_heads"], metric_for_best="eval_dummy_f1", greater_is_better=True, early_stopping=True, early_stopping_patience=2, num_workers=0, seed=7, logging_steps=4)` and a dummy `compute_metrics`.

multi-v1 output (exact, progress bars removed):

```text
precision before trainer True False
precision after trainer init False True model type PeftModel
train seconds 1.8 peak MiB 1563
summary keys {'total_steps': 12, 'total_epochs': 2, 'total_time_seconds': 1.7716383934020996, 'samples_per_second': 27.093565017985973, 'best_metric': 0.30000000000000004}
eval history [{'eval_loss': 23.7965, 'step': 4, 'eval_dummy_f1': 0.1}, {'eval_loss': 22.6076, 'step': 8, 'eval_dummy_f1': 0.2}, {'eval_loss': 22.2307, 'step': 12, 'eval_dummy_f1': 0.3}]
   best ['README.md', 'adapter_config.json', 'adapter_model.safetensors']
   checkpoint-12 ['README.md', 'adapter_config.json', 'adapter_model.safetensors']
   checkpoint-4 ['README.md', 'adapter_config.json', 'adapter_model.safetensors']
   checkpoint-8 ['README.md', 'adapter_config.json', 'adapter_model.safetensors']
   final ['README.md', 'adapter_config.json', 'adapter_model.safetensors']
   logs []
   training_config.json 
adapter_config ['input_projection', 'compat_mix', 'q_proj', 'endpoint_difference_projection', 'classifier.0'] ...
peft wrapped type PeftModel has extract_entities via getattr: True
{'entities': {'person': [{'text': 'Angela Merkel', 'confidence': 0.9966253042221069}], 'location': [{'text': 'Brussels', 'confidence': 0.998047947883606}]}}
merged type BoundaryExtractor
{'entities': {'person': [{'text': 'Angela Merkel', 'confidence': 0.9966309666633606}], 'location': [{'text': 'Brussels', 'confidence': 0.9980524778366089}]}}
```

base-v1 output (exact, same script):

```text
precision before trainer True False
precision after trainer init False True model type PeftModel
train seconds 2.3 peak MiB 1173
summary keys {'total_steps': 12, 'total_epochs': 2, 'total_time_seconds': 2.331428050994873, 'samples_per_second': 20.588239889932403, 'best_metric': 0.30000000000000004}
eval history [{'eval_loss': 22.4742, 'step': 4, 'eval_dummy_f1': 0.1}, {'eval_loss': 22.033, 'step': 8, 'eval_dummy_f1': 0.2}, {'eval_loss': 22.2977, 'step': 12, 'eval_dummy_f1': 0.3}]
   best ['README.md', 'adapter_config.json', 'adapter_model.safetensors']
   checkpoint-12 ['README.md', 'adapter_config.json', 'adapter_model.safetensors']
   checkpoint-4 ['README.md', 'adapter_config.json', 'adapter_model.safetensors']
   checkpoint-8 ['README.md', 'adapter_config.json', 'adapter_model.safetensors']
   final ['README.md', 'adapter_config.json', 'adapter_model.safetensors']
   logs []
   training_config.json 
adapter_config ['prior_projection', 'end_projection', 'input_projection', 'inside_weight', 'count_head'] ...
peft wrapped type PeftModel has extract_entities via getattr: True
{'entities': {'person': [{'text': 'Angela Merkel', 'confidence': 0.9970082640647888}], 'location': [{'text': 'Brussels', 'confidence': 0.9981801509857178}]}}
merged type BoundaryExtractor
{'entities': {'person': [{'text': 'Angela Merkel', 'confidence': 0.997010350227356}], 'location': [{'text': 'Brussels', 'confidence': 0.9981799125671387}]}}
```

Reading:

- Multi-task training (NER + classification + relation + structure in one set) runs. Eval at steps 4, 8, 12 works, so the old epoch-eval bug (issue #148) does not hit step eval.
- The trainer switched fp16 to bf16 for the boundary model.
- The saved PEFT `target_modules` list is short leaf names (for example `q_proj`, `classifier.0`). PEFT matches by name suffix on reload. It reloaded without error; I did not check that it matched no extra layers (unconfirmed).
- The dummy metric only tests the hook. 12 steps on 24 examples say nothing about quality.

### 7.6 Raw outputs, gliner2.5-multi-v1 (student)

Exact output of `handson2.py fastino/gliner2.5-multi-v1 12fc40399dae672ce840c5e3c50a92340bff3c8c`. Each block is `### <call> | <text>` then the JSON result.

```text
### env
{"gliner2": "2.0.0", "torch": "2.14.0+cu130", "model": "fastino/gliner2.5-multi-v1", "revision": "12fc40399dae672ce840c5e3c50a92340bff3c8c", "class": "BoundaryExtractor", "dtype": "torch.float32", "gpu": "NVIDIA GeForce RTX 3090", "load_seconds": 5.7, "gpu_mem_after_load_MiB": 1097}

### NER conll labels-only threshold=0.5 | EU rejects German call to boycott British lamb.
{"entities": {"person": [{"text": "British lamb", "confidence": 0.5734981298446655, "start": 34, "end": 46}], "organization": [{"text": "EU", "confidence": 0.9816170930862427, "start": 0, "end": 2}], "location": [], "miscellaneous": [{"text": "British lamb", "confidence": 0.6631917953491211, "start": 34, "end": 46}]}}

### NER conll labels-only threshold=0.01 | EU rejects German call to boycott British lamb.
{"entities": {"person": [{"text": "British lamb", "confidence": 0.5734981298446655, "start": 34, "end": 46}, {"text": "German", "confidence": 0.1995149403810501, "start": 11, "end": 17}, {"text": "EU", "confidence": 0.014091025106608868, "start": 0, "end": 2}], "organization": [{"text": "EU", "confidence": 0.9816170930862427, "start": 0, "end": 2}, {"text": "British lamb", "confidence": 0.05413828790187836, "start": 34, "end": 46}, {"text": "German", "confidence": 0.028922386467456818, "start": 11, "end": 17}, {"text": "call", "confidence": 0.01332993246614933, "start": 18, "end": 22}], "location": [], "miscellaneous": [{"text": "British lamb", "confidence": 0.6631917953491211, "start": 34, "end": 46}, {"text": "German", "confidence": 0.3604068458080292, "start": 11, "end": 17}, {"text": "EU", "confidence": 0.040665071457624435, "start": 0, "end": 2}, {"text": "call", "confidence": 0.017652546986937523, "start": 18, "end": 22}]}}

### NER conll with-descriptions threshold=0.5 | EU rejects German call to boycott British lamb.
{"entities": {"person": [], "organization": [{"text": "EU", "confidence": 0.9919707775115967, "start": 0, "end": 2}], "location": [], "miscellaneous": [{"text": "German", "confidence": 0.9450176358222961, "start": 11, "end": 17}, {"text": "British", "confidence": 0.7698947787284851, "start": 34, "end": 41}]}}

### NER conll threshold=0.0 candidate counts per label | EU rejects German call to boycott British lamb.
{"person": 8, "organization": 8, "location": 0, "miscellaneous": 8}

### NER conll labels-only threshold=0.5 | Peter Blackburn met officials of the European Commission in Brussels on Thursday.
{"entities": {"person": [{"text": "Peter Blackburn", "confidence": 0.9985833168029785, "start": 0, "end": 15}], "organization": [{"text": "European Commission", "confidence": 0.9937601685523987, "start": 37, "end": 56}], "location": [{"text": "Brussels", "confidence": 0.999358594417572, "start": 60, "end": 68}], "miscellaneous": []}}

### NER conll labels-only threshold=0.01 | Peter Blackburn met officials of the European Commission in Brussels on Thursday.
{"entities": {"person": [{"text": "Peter Blackburn", "confidence": 0.9985833168029785, "start": 0, "end": 15}], "organization": [{"text": "European Commission", "confidence": 0.9937601685523987, "start": 37, "end": 56}], "location": [{"text": "Brussels", "confidence": 0.999358594417572, "start": 60, "end": 68}], "miscellaneous": []}}

### NER conll with-descriptions threshold=0.5 | Peter Blackburn met officials of the European Commission in Brussels on Thursday.
{"entities": {"person": [{"text": "Peter Blackburn", "confidence": 0.9989184141159058, "start": 0, "end": 15}], "organization": [{"text": "European Commission", "confidence": 0.9964742064476013, "start": 37, "end": 56}], "location": [{"text": "Brussels", "confidence": 0.9993649125099182, "start": 60, "end": 68}], "miscellaneous": []}}

### NER conll threshold=0.0 candidate counts per label | Peter Blackburn met officials of the European Commission in Brussels on Thursday.
{"person": 11, "organization": 11, "location": 11, "miscellaneous": 0}

### NER conll labels-only threshold=0.5 | Emmanuel Macron a rencontré les dirigeants de Renault à Paris mardi.
{"entities": {"person": [{"text": "Emmanuel Macron", "confidence": 0.998593270778656, "start": 0, "end": 15}], "organization": [{"text": "Renault", "confidence": 0.9949976205825806, "start": 46, "end": 53}], "location": [{"text": "Paris", "confidence": 0.9987353682518005, "start": 56, "end": 61}], "miscellaneous": []}}

### NER conll labels-only threshold=0.01 | Emmanuel Macron a rencontré les dirigeants de Renault à Paris mardi.
{"entities": {"person": [{"text": "Emmanuel Macron", "confidence": 0.998593270778656, "start": 0, "end": 15}], "organization": [{"text": "Renault", "confidence": 0.9949976205825806, "start": 46, "end": 53}], "location": [{"text": "Paris", "confidence": 0.9987353682518005, "start": 56, "end": 61}], "miscellaneous": []}}

### NER conll with-descriptions threshold=0.5 | Emmanuel Macron a rencontré les dirigeants de Renault à Paris mardi.
{"entities": {"person": [{"text": "Emmanuel Macron", "confidence": 0.9989646673202515, "start": 0, "end": 15}], "organization": [{"text": "Renault", "confidence": 0.9966073036193848, "start": 46, "end": 53}], "location": [{"text": "Paris", "confidence": 0.9984802603721619, "start": 56, "end": 61}], "miscellaneous": []}}

### NER conll threshold=0.0 candidate counts per label | Emmanuel Macron a rencontré les dirigeants de Renault à Paris mardi.
{"person": 11, "organization": 11, "location": 11, "miscellaneous": 0}

### NER bc5cdr labels-only threshold=0.5 | Naloxone reverses the antihypertensive effect of clonidine in patients with essential hypertension.
{"entities": {"chemical": [{"text": "clonidine", "confidence": 0.9845027327537537, "start": 49, "end": 58}, {"text": "Naloxone", "confidence": 0.9801826477050781, "start": 0, "end": 8}], "disease": [{"text": "essential hypertension", "confidence": 0.9967663288116455, "start": 76, "end": 98}]}}

### NER bc5cdr labels-only threshold=0.01 | Naloxone reverses the antihypertensive effect of clonidine in patients with essential hypertension.
{"entities": {"chemical": [{"text": "clonidine", "confidence": 0.9845027327537537, "start": 49, "end": 58}, {"text": "Naloxone", "confidence": 0.9801826477050781, "start": 0, "end": 8}], "disease": [{"text": "essential hypertension", "confidence": 0.9967663288116455, "start": 76, "end": 98}]}}

### NER bc5cdr with-descriptions threshold=0.5 | Naloxone reverses the antihypertensive effect of clonidine in patients with essential hypertension.
{"entities": {"chemical": [{"text": "clonidine", "confidence": 0.9820658564567566, "start": 49, "end": 58}, {"text": "Naloxone", "confidence": 0.9773633480072021, "start": 0, "end": 8}], "disease": [{"text": "essential hypertension", "confidence": 0.9967418313026428, "start": 76, "end": 98}]}}

### NER bc5cdr threshold=0.0 candidate counts per label | Naloxone reverses the antihypertensive effect of clonidine in patients with essential hypertension.
{"chemical": 12, "disease": 11}

### NER bc5cdr labels-only threshold=0.5 | Lithium-induced nephrogenic diabetes insipidus was observed in two patients.
{"entities": {"chemical": [], "disease": [{"text": "nephrogenic diabetes insipidus", "confidence": 0.9912112355232239, "start": 16, "end": 46}]}}

### NER bc5cdr labels-only threshold=0.01 | Lithium-induced nephrogenic diabetes insipidus was observed in two patients.
{"entities": {"chemical": [], "disease": [{"text": "nephrogenic diabetes insipidus", "confidence": 0.9912112355232239, "start": 16, "end": 46}, {"text": "patients", "confidence": 0.010839507915079594, "start": 67, "end": 75}]}}

### NER bc5cdr with-descriptions threshold=0.5 | Lithium-induced nephrogenic diabetes insipidus was observed in two patients.
{"entities": {"chemical": [], "disease": [{"text": "Lithium-induced nephrogenic diabetes insipidus", "confidence": 0.9461052417755127, "start": 0, "end": 46}]}}

### NER bc5cdr threshold=0.0 candidate counts per label | Lithium-induced nephrogenic diabetes insipidus was observed in two patients.
{"chemical": 0, "disease": 8}

### NER mit_movie labels-only threshold=0.5 | show me a comedy directed by wes anderson from the 1990s starring bill murray
{"entities": {"genre": [{"text": "comedy", "confidence": 0.9791254997253418, "start": 10, "end": 16}], "director": [{"text": "wes anderson", "confidence": 0.9922975897789001, "start": 29, "end": 41}], "year": [{"text": "1990s", "confidence": 0.9916799664497375, "start": 51, "end": 56}], "actor": [{"text": "bill murray", "confidence": 0.9983091354370117, "start": 66, "end": 77}], "title": []}}

### NER mit_movie labels-only threshold=0.01 | show me a comedy directed by wes anderson from the 1990s starring bill murray
{"entities": {"genre": [{"text": "comedy", "confidence": 0.9791254997253418, "start": 10, "end": 16}], "director": [{"text": "wes anderson", "confidence": 0.9922975897789001, "start": 29, "end": 41}], "year": [{"text": "1990s", "confidence": 0.9916799664497375, "start": 51, "end": 56}], "actor": [{"text": "bill murray", "confidence": 0.9983091354370117, "start": 66, "end": 77}], "title": [{"text": "show me a comedy", "confidence": 0.43278029561042786, "start": 0, "end": 16}, {"text": "bill murray", "confidence": 0.013377207331359386, "start": 66, "end": 77}]}}

### NER mit_movie with-descriptions threshold=0.5 | show me a comedy directed by wes anderson from the 1990s starring bill murray
{"entities": {"genre": [{"text": "comedy", "confidence": 0.9712996482849121, "start": 10, "end": 16}], "director": [{"text": "wes anderson", "confidence": 0.9951298236846924, "start": 29, "end": 41}], "year": [{"text": "1990s", "confidence": 0.9945188164710999, "start": 51, "end": 56}], "actor": [{"text": "bill murray", "confidence": 0.9992315769195557, "start": 66, "end": 77}], "title": [{"text": "show me a comedy", "confidence": 0.648583710193634, "start": 0, "end": 16}]}}

### NER mit_movie threshold=0.0 candidate counts per label | show me a comedy directed by wes anderson from the 1990s starring bill murray
{"genre": 15, "director": 13, "year": 15, "actor": 13, "title": 10}

### REL threshold=0.5 | Geoffrey Hinton, a professor at the University of Toronto, won the Turing Award in 2018 together with Yann LeCun.
{"relation_extraction": {"role": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.880390465259552}, "tail": {"text": "professor", "start": 19, "end": 28, "confidence": 0.880390465259552}}], "win-defeat": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.7487406730651855}, "tail": {"text": "Turing Award", "start": 67, "end": 79, "confidence": 0.7487406730651855}}], "general-affiliation": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.8442524671554565}, "tail": {"text": "University of Toronto", "start": 36, "end": 57, "confidence": 0.8442524671554565}}], "origin": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.5729935765266418}, "tail": {"text": "University of Toronto", "start": 36, "end": 57, "confidence": 0.5729935765266418}}], "physical": [], "temporal": []}}

### REL threshold=0.01 | Geoffrey Hinton, a professor at the University of Toronto, won the Turing Award in 2018 together with Yann LeCun.
{"relation_extraction": {"role": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.880390465259552}, "tail": {"text": "professor", "start": 19, "end": 28, "confidence": 0.880390465259552}}], "win-defeat": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.7487406730651855}, "tail": {"text": "Turing Award", "start": 67, "end": 79, "confidence": 0.7487406730651855}}], "general-affiliation": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.8442524671554565}, "tail": {"text": "University of Toronto", "start": 36, "end": 57, "confidence": 0.8442524671554565}}], "origin": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.5729935765266418}, "tail": {"text": "University of Toronto", "start": 36, "end": 57, "confidence": 0.5729935765266418}}], "temporal": [{"head": {"text": "Turing Award", "start": 67, "end": 79, "confidence": 0.45213446021080017}, "tail": {"text": "2018", "start": 83, "end": 87, "confidence": 0.45213446021080017}}], "physical": []}}

### REL threshold=0.0 | Geoffrey Hinton, a professor at the University of Toronto, won the Turing Award in 2018 together with Yann LeCun.
{"relation_extraction": {"role": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.880390465259552}, "tail": {"text": "professor", "start": 19, "end": 28, "confidence": 0.880390465259552}}], "win-defeat": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.7487406730651855}, "tail": {"text": "Turing Award", "start": 67, "end": 79, "confidence": 0.7487406730651855}}], "general-affiliation": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.8442524671554565}, "tail": {"text": "University of Toronto", "start": 36, "end": 57, "confidence": 0.8442524671554565}}], "origin": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.5729935765266418}, "tail": {"text": "University of Toronto", "start": 36, "end": 57, "confidence": 0.5729935765266418}}], "temporal": [{"head": {"text": "Turing Award", "start": 67, "end": 79, "confidence": 0.45213446021080017}, "tail": {"text": "2018", "start": 83, "end": 87, "confidence": 0.45213446021080017}}], "physical": []}}

### REL threshold=0.5 | Marie Curie was born in Warsaw and later worked at the University of Paris.
{"relation_extraction": {"general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.8147777915000916}, "tail": {"text": "University of Paris", "start": 55, "end": 74, "confidence": 0.8147777915000916}}], "origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9735365509986877}, "tail": {"text": "Warsaw", "start": 24, "end": 30, "confidence": 0.9735365509986877}}], "role": [], "win-defeat": [], "physical": [], "temporal": []}}

### REL threshold=0.01 | Marie Curie was born in Warsaw and later worked at the University of Paris.
{"relation_extraction": {"general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.8147777915000916}, "tail": {"text": "University of Paris", "start": 55, "end": 74, "confidence": 0.8147777915000916}}], "origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9735365509986877}, "tail": {"text": "Warsaw", "start": 24, "end": 30, "confidence": 0.9735365509986877}}], "role": [], "win-defeat": [], "physical": [], "temporal": []}}

### REL threshold=0.0 | Marie Curie was born in Warsaw and later worked at the University of Paris.
{"relation_extraction": {"general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.8147777915000916}, "tail": {"text": "University of Paris", "start": 55, "end": 74, "confidence": 0.8147777915000916}}], "origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9735365509986877}, "tail": {"text": "Warsaw", "start": 24, "end": 30, "confidence": 0.9735365509986877}}], "role": [], "win-defeat": [], "physical": [], "temporal": []}}

### REL threshold=0.5 | Marie Curie est née à Varsovie et a ensuite travaillé à l'Université de Paris.
{"relation_extraction": {"general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.6321074366569519}, "tail": {"text": "Université de Paris", "start": 58, "end": 77, "confidence": 0.6321074366569519}}], "origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9252339601516724}, "tail": {"text": "Varsovie", "start": 22, "end": 30, "confidence": 0.9252339601516724}}], "role": [], "win-defeat": [], "physical": [], "temporal": []}}

### REL threshold=0.01 | Marie Curie est née à Varsovie et a ensuite travaillé à l'Université de Paris.
{"relation_extraction": {"general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.6321074366569519}, "tail": {"text": "Université de Paris", "start": 58, "end": 77, "confidence": 0.6321074366569519}}], "origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9252339601516724}, "tail": {"text": "Varsovie", "start": 22, "end": 30, "confidence": 0.9252339601516724}}], "role": [], "win-defeat": [], "physical": [], "temporal": []}}

### REL threshold=0.0 | Marie Curie est née à Varsovie et a ensuite travaillé à l'Université de Paris.
{"relation_extraction": {"general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.6321074366569519}, "tail": {"text": "Université de Paris", "start": 58, "end": 77, "confidence": 0.6321074366569519}}], "origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9252339601516724}, "tail": {"text": "Varsovie", "start": 22, "end": 30, "confidence": 0.9252339601516724}}], "role": [], "win-defeat": [], "physical": [], "temporal": []}}

### REL with-descriptions threshold=0.5 | Marie Curie was born in Warsaw and later worked at the University of Paris.
{"relation_extraction": {"origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.8895017504692078}, "tail": {"text": "Warsaw", "start": 24, "end": 30, "confidence": 0.8895017504692078}}], "general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.5783007740974426}, "tail": {"text": "University of Paris", "start": 55, "end": 74, "confidence": 0.5783007740974426}}], "role": [], "win-defeat": []}}

### CLS multi_label cls_threshold=0.5 | Overexpression of VEGF promoted tumor angiogenesis and increased the metastatic potential of breast cancer cells.
{"hallmarks": [{"label": "inducing angiogenesis", "confidence": 0.932393491268158}]}

### CLS multi_label cls_threshold=0.0 | Overexpression of VEGF promoted tumor angiogenesis and increased the metastatic potential of breast cancer cells.
{"hallmarks": [{"label": "sustaining proliferative signaling", "confidence": 0.028346214443445206}, {"label": "evading growth suppressors", "confidence": 0.0014603142626583576}, {"label": "resisting cell death", "confidence": 0.007821178063750267}, {"label": "enabling replicative immortality", "confidence": 0.008600751869380474}, {"label": "inducing angiogenesis", "confidence": 0.932393491268158}, {"label": "activating invasion and metastasis", "confidence": 0.019743887707591057}, {"label": "genomic instability and mutation", "confidence": 0.007317432202398777}, {"label": "tumor promoting inflammation", "confidence": 0.11621269583702087}, {"label": "deregulating cellular energetics", "confidence": 0.0025878241285681725}, {"label": "avoiding immune destruction", "confidence": 0.002725249622017145}]}

### CLS multi_label cls_threshold=0.5 | Loss of p53 function allowed cells to escape apoptosis and accumulate chromosomal aberrations.
{"hallmarks": [{"label": "resisting cell death", "confidence": 0.4277968108654022}]}

### CLS multi_label cls_threshold=0.0 | Loss of p53 function allowed cells to escape apoptosis and accumulate chromosomal aberrations.
{"hallmarks": [{"label": "sustaining proliferative signaling", "confidence": 0.08880899101495743}, {"label": "evading growth suppressors", "confidence": 0.24035371840000153}, {"label": "resisting cell death", "confidence": 0.4277968108654022}, {"label": "enabling replicative immortality", "confidence": 0.05218016728758812}, {"label": "inducing angiogenesis", "confidence": 0.06508969515562057}, {"label": "activating invasion and metastasis", "confidence": 0.02987240068614483}, {"label": "genomic instability and mutation", "confidence": 0.29880526661872864}, {"label": "tumor promoting inflammation", "confidence": 0.03813092038035393}, {"label": "deregulating cellular energetics", "confidence": 0.05454038083553314}, {"label": "avoiding immune destruction", "confidence": 0.12060019373893738}]}

### CLS multi_label cls_threshold=0.5 | Patients were recruited from three hospitals between 2005 and 2010.
{"hallmarks": [{"label": "sustaining proliferative signaling", "confidence": 0.17933155596256256}]}

### CLS multi_label cls_threshold=0.0 | Patients were recruited from three hospitals between 2005 and 2010.
{"hallmarks": [{"label": "sustaining proliferative signaling", "confidence": 0.17933155596256256}, {"label": "evading growth suppressors", "confidence": 0.012355735525488853}, {"label": "resisting cell death", "confidence": 0.05583095923066139}, {"label": "enabling replicative immortality", "confidence": 0.06566415727138519}, {"label": "inducing angiogenesis", "confidence": 0.11467190086841583}, {"label": "activating invasion and metastasis", "confidence": 0.025575116276741028}, {"label": "genomic instability and mutation", "confidence": 0.040664780884981155}, {"label": "tumor promoting inflammation", "confidence": 0.052024852484464645}, {"label": "deregulating cellular energetics", "confidence": 0.017073366791009903}, {"label": "avoiding immune destruction", "confidence": 0.03157336264848709}]}

### CLS multi_label cls_threshold=0.5 | La surexpression du VEGF a favorisé l'angiogenèse tumorale et augmenté le potentiel métastatique des cellules.
{"hallmarks": [{"label": "inducing angiogenesis", "confidence": 0.7132778763771057}]}

### CLS multi_label cls_threshold=0.0 | La surexpression du VEGF a favorisé l'angiogenèse tumorale et augmenté le potentiel métastatique des cellules.
{"hallmarks": [{"label": "sustaining proliferative signaling", "confidence": 0.03826800733804703}, {"label": "evading growth suppressors", "confidence": 0.0037954598665237427}, {"label": "resisting cell death", "confidence": 0.014569059014320374}, {"label": "enabling replicative immortality", "confidence": 0.01581631973385811}, {"label": "inducing angiogenesis", "confidence": 0.7132778763771057}, {"label": "activating invasion and metastasis", "confidence": 0.06220089644193649}, {"label": "genomic instability and mutation", "confidence": 0.02017829939723015}, {"label": "tumor promoting inflammation", "confidence": 0.2524065375328064}, {"label": "deregulating cellular energetics", "confidence": 0.007749286014586687}, {"label": "avoiding immune destruction", "confidence": 0.0050645978190004826}]}

### CLS Classifier.classify (all-label probabilities) | Overexpression of VEGF promoted tumor angiogenesis and increased the metastatic potential of breast cancer cells.
{"hallmarks": {"value": ["inducing angiogenesis"], "confidence": 0.9730836690664496, "probabilities": {"sustaining proliferative signaling": 0.028346213860931797, "evading growth suppressors": 0.0014603142428367326, "resisting cell death": 0.007821178385594358, "enabling replicative immortality": 0.008600751681800042, "inducing angiogenesis": 0.9323934732218379, "activating invasion and metastasis": 0.019743887430038357, "genomic instability and mutation": 0.007317432108044672, "tumor promoting inflammation": 0.11621270014640563, "deregulating cellular energetics": 0.00258782419573867, "avoiding immune destruction": 0.0027252495775191943}}, "_meta": {"feasible": true, "decoder": "independent", "exact": true, "objective": 2.6240503787994385, "violations": []}}

### CLS Classifier.classify (all-label probabilities) | Loss of p53 function allowed cells to escape apoptosis and accumulate chromosomal aberrations.
{"hallmarks": {"value": [], "confidence": null, "probabilities": {"sustaining proliferative signaling": 0.08880898587894906, "evading growth suppressors": 0.24035370055609304, "resisting cell death": 0.42779683294855175, "enabling replicative immortality": 0.05218016550724949, "inducing angiogenesis": 0.06508969845631817, "activating invasion and metastasis": 0.029872401165116424, "genomic instability and mutation": 0.2988052698106716, "tumor promoting inflammation": 0.038130920990887295, "deregulating cellular energetics": 0.05454038193394823, "avoiding immune destruction": 0.12060019763480019}}, "_meta": {"feasible": true, "decoder": "independent", "exact": true, "objective": 0.0, "violations": []}}

### CLS Classifier.classify (all-label probabilities) | Patients were recruited from three hospitals between 2005 and 2010.
{"hallmarks": {"value": [], "confidence": null, "probabilities": {"sustaining proliferative signaling": 0.17933155279225216, "evading growth suppressors": 0.01235573534865887, "resisting cell death": 0.055830957893381696, "enabling replicative immortality": 0.06566415648793564, "inducing angiogenesis": 0.11467189795920082, "activating invasion and metastasis": 0.025575116764102804, "genomic instability and mutation": 0.04066478087451065, "tumor promoting inflammation": 0.05202485365409939, "deregulating cellular energetics": 0.01707336636760382, "avoiding immune destruction": 0.03157336274213146}}, "_meta": {"feasible": true, "decoder": "independent", "exact": true, "objective": 0.0, "violations": []}}

### CLS Classifier.classify (all-label probabilities) | La surexpression du VEGF a favorisé l'angiogenèse tumorale et augmenté le potentiel métastatique des cellules.
{"hallmarks": {"value": ["inducing angiogenesis"], "confidence": 0.9231456685560309, "probabilities": {"sustaining proliferative signaling": 0.038268007992691616, "evading growth suppressors": 0.003795459791351183, "resisting cell death": 0.014569058665327643, "enabling replicative immortality": 0.015816320334797684, "inducing angiogenesis": 0.713277897781197, "activating invasion and metastasis": 0.0622009023910766, "genomic instability and mutation": 0.02017829877117173, "tumor promoting inflammation": 0.2524065391047752, "deregulating cellular energetics": 0.0077492865431152596, "avoiding immune destruction": 0.005064597540918297}}, "_meta": {"feasible": true, "decoder": "independent", "exact": true, "objective": 0.9113576412200928, "violations": []}}

### JSON extract_json (record mode) threshold=0.5 | wake me up at 7 am tomorrow
{"slots": [{"time": [{"text": "7 am", "confidence": 0.9757475256919861, "start": 14, "end": 18}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON extract_json (record mode) threshold=0.01 | wake me up at 7 am tomorrow
{"slots": [{"time": [{"text": "at 7 am", "confidence": 0.04174575209617615, "start": 11, "end": 18}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "7", "confidence": 0.011940461583435535, "start": 14, "end": 15}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "7 am", "confidence": 0.9757475256919861, "start": 14, "end": 18}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "7 am tomorrow", "confidence": 0.035229865461587906, "start": 14, "end": 27}], "date": [{"text": "tomorrow", "confidence": 0.7713441848754883, "start": 19, "end": 27}], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON Schema.structure(mode=None) threshold=0.5 | wake me up at 7 am tomorrow
{"slots": [{"time": [{"text": "7 am", "confidence": 0.9757475256919861, "start": 14, "end": 18}], "date": [{"text": "tomorrow", "confidence": 0.8862467408180237, "start": 19, "end": 27}], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON-as-entities threshold=0.5 | wake me up at 7 am tomorrow
{"entities": {"time": [{"text": "7 am", "confidence": 0.9957976341247559, "start": 14, "end": 18}], "date": [{"text": "tomorrow", "confidence": 0.9876790642738342, "start": 19, "end": 27}], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}}

### JSON extract_json (record mode) threshold=0.5 | play the latest album by taylor swift on spotify
{}

### JSON extract_json (record mode) threshold=0.01 | play the latest album by taylor swift on spotify
{}

### JSON Schema.structure(mode=None) threshold=0.5 | play the latest album by taylor swift on spotify
{"slots": [{"time": [], "date": [], "artist_name": [{"text": "taylor swift", "confidence": 0.9870136380195618, "start": 25, "end": 37}], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON-as-entities threshold=0.5 | play the latest album by taylor swift on spotify
{"entities": {"time": [], "date": [], "artist_name": [{"text": "taylor swift", "confidence": 0.9978533387184143, "start": 25, "end": 37}], "app_name": [{"text": "spotify", "confidence": 0.9792935252189636, "start": 41, "end": 48}], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}}

### JSON extract_json (record mode) threshold=0.5 | what is the weather like
{}

### JSON extract_json (record mode) threshold=0.01 | what is the weather like
{}

### JSON Schema.structure(mode=None) threshold=0.5 | what is the weather like
{}

### JSON-as-entities threshold=0.5 | what is the weather like
{"entities": {"time": [], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [{"text": "weather", "confidence": 0.5379512906074524, "start": 12, "end": 19}], "media_type": []}}

### JSON extract_json (record mode) threshold=0.5 | olly tell me a joke
{}

### JSON extract_json (record mode) threshold=0.01 | olly tell me a joke
{}

### JSON Schema.structure(mode=None) threshold=0.5 | olly tell me a joke
{}

### JSON-as-entities threshold=0.5 | olly tell me a joke
{"entities": {"time": [], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}}

### JSON extract_json (record mode) threshold=0.5 | réveille-moi à 7 heures demain
{"slots": [{"time": [{"text": "7 heures", "confidence": 0.9749667644500732, "start": 15, "end": 23}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON extract_json (record mode) threshold=0.01 | réveille-moi à 7 heures demain
{"slots": [{"time": [{"text": "à 7 heures", "confidence": 0.030323930084705353, "start": 13, "end": 23}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "7", "confidence": 0.043142929673194885, "start": 15, "end": 16}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "7 heures", "confidence": 0.9749667644500732, "start": 15, "end": 23}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "7 heures demain", "confidence": 0.046636343002319336, "start": 15, "end": 30}], "date": [{"text": "demain", "confidence": 0.4957660734653473, "start": 24, "end": 30}], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON Schema.structure(mode=None) threshold=0.5 | réveille-moi à 7 heures demain
{"slots": [{"time": [{"text": "7 heures", "confidence": 0.9749667644500732, "start": 15, "end": 23}], "date": [{"text": "demain", "confidence": 0.789249062538147, "start": 24, "end": 30}], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON-as-entities threshold=0.5 | réveille-moi à 7 heures demain
{"entities": {"time": [{"text": "7 heures", "confidence": 0.9925011992454529, "start": 15, "end": 23}], "date": [{"text": "demain", "confidence": 0.9737963676452637, "start": 24, "end": 30}], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}}

### COMBINED threshold=0.5 | Geoffrey Hinton, a professor at the University of Toronto, won the Turing Award in 2018 together with Yann LeCun.
{"event": [{"winner": [{"text": "Geoffrey Hinton", "confidence": 0.9972794651985168, "start": 0, "end": 15}], "prize": {"text": "Turing Award", "confidence": 0.9963557720184326, "start": 67, "end": 79}, "year": {"text": "2018", "confidence": 0.9989244341850281, "start": 83, "end": 87}}], "entities": {"person": [{"text": "Geoffrey Hinton", "confidence": 0.9985129237174988, "start": 0, "end": 15}, {"text": "Yann LeCun", "confidence": 0.9972870349884033, "start": 102, "end": 112}], "organization": [{"text": "University of Toronto", "confidence": 0.9988893866539001, "start": 36, "end": 57}], "award": [{"text": "Turing Award", "confidence": 0.9983490705490112, "start": 67, "end": 79}]}, "topic": {"label": "science", "confidence": 0.8932009339332581}, "relation_extraction": {"win-defeat": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.9198168516159058}, "tail": {"text": "Turing Award", "start": 67, "end": 79, "confidence": 0.9198168516159058}}], "role": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.8965836763381958}, "tail": {"text": "professor", "start": 19, "end": 28, "confidence": 0.8965836763381958}}]}}

### LoRA target resolution
{"encoder": {"n_linear": 72, "top_level": ["encoder"]}, "encoder+span_rep+classifier+count_embed+count_pred": {"n_linear": 74, "top_level": ["classifier", "encoder"]}, "encoder+all_task_heads": {"n_linear": 131, "top_level": ["boundary_head", "classifier", "encoder", "record_decoder", "relation_scorer"]}, "all_task_heads": {"n_linear": 59, "top_level": ["boundary_head", "classifier", "record_decoder", "relation_scorer"]}, "encoder_leaf_names": ["dense", "key_proj", "query_proj", "value_proj"], "example_encoder_module": "encoder.encoder.layer.0.attention.output.dense", "params_total": 287355159, "params_by_module": {"encoder": 277536768, "classifier": 1182721, "boundary_head": 1619986, "record_decoder": 1108994, "relation_scorer": 5906690}}

### timing (RTX 3090, fp32, 5 repeats after warm-up)
{"ner_single_ms": 9.23, "rel_single_ms": 11.97, "cls10_single_ms": 7.02, "json_single_ms": 11.94, "combined_single_ms": 11.31, "ner_batch8_ms_per_sentence": 1.49, "ner_batch32_ms_per_sentence": 1.0, "cls10_batch32_ms_per_sentence": 1.97, "gpu_peak_mem_MiB": 1546}
```

### 7.7 Raw outputs, gliner2.5-base-v1 (comparison)

Exact output of the same script on the base snapshot folder. An earlier base run without the French sentences gave the same numbers for the shared sentences.

```text
### env
{"gliner2": "2.0.0", "torch": "2.14.0+cu130", "model": "/home/abhishek/.cache/huggingface/hub/models--fastino--gliner2.5-base-v1/snapshots/1a8bc24e00dc7300b9017c81d63e3dcdabb26596", "revision": "1a8bc24e00dc7300b9017c81d63e3dcdabb26596", "class": "BoundaryExtractor", "dtype": "torch.float32", "gpu": "NVIDIA GeForce RTX 3090", "load_seconds": 1.9, "gpu_mem_after_load_MiB": 740}

### NER conll labels-only threshold=0.5 | EU rejects German call to boycott British lamb.
{"entities": {"person": [{"text": "British lamb", "confidence": 0.6599313020706177, "start": 34, "end": 46}], "organization": [{"text": "EU", "confidence": 0.9946703314781189, "start": 0, "end": 2}], "location": [{"text": "German", "confidence": 0.7222973704338074, "start": 11, "end": 17}, {"text": "British", "confidence": 0.565673828125, "start": 34, "end": 41}], "miscellaneous": []}}

### NER conll labels-only threshold=0.01 | EU rejects German call to boycott British lamb.
{"entities": {"person": [{"text": "British lamb", "confidence": 0.6599313020706177, "start": 34, "end": 46}, {"text": "German", "confidence": 0.2168760597705841, "start": 11, "end": 17}], "organization": [{"text": "EU", "confidence": 0.9946703314781189, "start": 0, "end": 2}, {"text": "German", "confidence": 0.03630439192056656, "start": 11, "end": 17}, {"text": "British", "confidence": 0.03136632964015007, "start": 34, "end": 41}], "location": [{"text": "German", "confidence": 0.7222973704338074, "start": 11, "end": 17}, {"text": "British", "confidence": 0.565673828125, "start": 34, "end": 41}], "miscellaneous": []}}

### NER conll with-descriptions threshold=0.5 | EU rejects German call to boycott British lamb.
{"entities": {"person": [], "organization": [{"text": "EU", "confidence": 0.9959228038787842, "start": 0, "end": 2}], "location": [], "miscellaneous": [{"text": "British lamb", "confidence": 0.710374116897583, "start": 34, "end": 46}, {"text": "German", "confidence": 0.6756882071495056, "start": 11, "end": 17}]}}

### NER conll threshold=0.0 candidate counts per label | EU rejects German call to boycott British lamb.
{"person": 8, "organization": 9, "location": 9, "miscellaneous": 0}

### NER conll labels-only threshold=0.5 | Peter Blackburn met officials of the European Commission in Brussels on Thursday.
{"entities": {"person": [{"text": "Peter Blackburn", "confidence": 0.9977014660835266, "start": 0, "end": 15}], "organization": [{"text": "European Commission", "confidence": 0.9916440844535828, "start": 37, "end": 56}], "location": [{"text": "Brussels", "confidence": 0.996350884437561, "start": 60, "end": 68}], "miscellaneous": []}}

### NER conll labels-only threshold=0.01 | Peter Blackburn met officials of the European Commission in Brussels on Thursday.
{"entities": {"person": [{"text": "Peter Blackburn", "confidence": 0.9977014660835266, "start": 0, "end": 15}], "organization": [{"text": "European Commission", "confidence": 0.9916440844535828, "start": 37, "end": 56}], "location": [{"text": "Brussels", "confidence": 0.996350884437561, "start": 60, "end": 68}], "miscellaneous": []}}

### NER conll with-descriptions threshold=0.5 | Peter Blackburn met officials of the European Commission in Brussels on Thursday.
{"entities": {"person": [{"text": "Peter Blackburn", "confidence": 0.9989399313926697, "start": 0, "end": 15}], "organization": [{"text": "European Commission", "confidence": 0.9956379532814026, "start": 37, "end": 56}], "location": [{"text": "Brussels", "confidence": 0.9977849125862122, "start": 60, "end": 68}], "miscellaneous": []}}

### NER conll threshold=0.0 candidate counts per label | Peter Blackburn met officials of the European Commission in Brussels on Thursday.
{"person": 12, "organization": 11, "location": 12, "miscellaneous": 0}

### NER conll labels-only threshold=0.5 | Emmanuel Macron a rencontré les dirigeants de Renault à Paris mardi.
{"entities": {"person": [{"text": "Emmanuel Macron", "confidence": 0.9947609305381775, "start": 0, "end": 15}], "organization": [{"text": "Renault", "confidence": 0.9820849299430847, "start": 46, "end": 53}], "location": [{"text": "Paris", "confidence": 0.9942513704299927, "start": 56, "end": 61}], "miscellaneous": []}}

### NER conll labels-only threshold=0.01 | Emmanuel Macron a rencontré les dirigeants de Renault à Paris mardi.
{"entities": {"person": [{"text": "Emmanuel Macron", "confidence": 0.9947609305381775, "start": 0, "end": 15}, {"text": "Renault", "confidence": 0.023890547454357147, "start": 46, "end": 53}, {"text": "dirigeants", "confidence": 0.021290432661771774, "start": 32, "end": 42}], "organization": [{"text": "Renault", "confidence": 0.9820849299430847, "start": 46, "end": 53}], "location": [{"text": "Paris", "confidence": 0.9942513704299927, "start": 56, "end": 61}], "miscellaneous": []}}

### NER conll with-descriptions threshold=0.5 | Emmanuel Macron a rencontré les dirigeants de Renault à Paris mardi.
{"entities": {"person": [{"text": "Emmanuel Macron", "confidence": 0.9962132573127747, "start": 0, "end": 15}], "organization": [{"text": "Renault", "confidence": 0.9913235902786255, "start": 46, "end": 53}], "location": [{"text": "Paris", "confidence": 0.9959805011749268, "start": 56, "end": 61}], "miscellaneous": []}}

### NER conll threshold=0.0 candidate counts per label | Emmanuel Macron a rencontré les dirigeants de Renault à Paris mardi.
{"person": 11, "organization": 12, "location": 12, "miscellaneous": 0}

### NER bc5cdr labels-only threshold=0.5 | Naloxone reverses the antihypertensive effect of clonidine in patients with essential hypertension.
{"entities": {"chemical": [{"text": "clonidine", "confidence": 0.9985668063163757, "start": 49, "end": 58}, {"text": "Naloxone", "confidence": 0.9973691701889038, "start": 0, "end": 8}], "disease": [{"text": "essential hypertension", "confidence": 0.9957183003425598, "start": 76, "end": 98}]}}

### NER bc5cdr labels-only threshold=0.01 | Naloxone reverses the antihypertensive effect of clonidine in patients with essential hypertension.
{"entities": {"chemical": [{"text": "clonidine", "confidence": 0.9985668063163757, "start": 49, "end": 58}, {"text": "Naloxone", "confidence": 0.9973691701889038, "start": 0, "end": 8}], "disease": [{"text": "essential hypertension", "confidence": 0.9957183003425598, "start": 76, "end": 98}]}}

### NER bc5cdr with-descriptions threshold=0.5 | Naloxone reverses the antihypertensive effect of clonidine in patients with essential hypertension.
{"entities": {"chemical": [{"text": "clonidine", "confidence": 0.9992563128471375, "start": 49, "end": 58}, {"text": "Naloxone", "confidence": 0.9984802603721619, "start": 0, "end": 8}], "disease": [{"text": "essential hypertension", "confidence": 0.9959262609481812, "start": 76, "end": 98}]}}

### NER bc5cdr threshold=0.0 candidate counts per label | Naloxone reverses the antihypertensive effect of clonidine in patients with essential hypertension.
{"chemical": 12, "disease": 12}

### NER bc5cdr labels-only threshold=0.5 | Lithium-induced nephrogenic diabetes insipidus was observed in two patients.
{"entities": {"chemical": [{"text": "Lithium-induced", "confidence": 0.7477558851242065, "start": 0, "end": 15}], "disease": [{"text": "nephrogenic diabetes insipidus", "confidence": 0.8581393957138062, "start": 16, "end": 46}]}}

### NER bc5cdr labels-only threshold=0.01 | Lithium-induced nephrogenic diabetes insipidus was observed in two patients.
{"entities": {"chemical": [{"text": "Lithium-induced", "confidence": 0.7477558851242065, "start": 0, "end": 15}], "disease": [{"text": "nephrogenic diabetes insipidus", "confidence": 0.8581393957138062, "start": 16, "end": 46}]}}

### NER bc5cdr with-descriptions threshold=0.5 | Lithium-induced nephrogenic diabetes insipidus was observed in two patients.
{"entities": {"chemical": [{"text": "Lithium-induced", "confidence": 0.6307183504104614, "start": 0, "end": 15}], "disease": [{"text": "nephrogenic diabetes insipidus", "confidence": 0.8468292355537415, "start": 16, "end": 46}]}}

### NER bc5cdr threshold=0.0 candidate counts per label | Lithium-induced nephrogenic diabetes insipidus was observed in two patients.
{"chemical": 10, "disease": 8}

### NER mit_movie labels-only threshold=0.5 | show me a comedy directed by wes anderson from the 1990s starring bill murray
{"entities": {"genre": [{"text": "comedy", "confidence": 0.9873633980751038, "start": 10, "end": 16}], "director": [{"text": "wes anderson", "confidence": 0.9990099668502808, "start": 29, "end": 41}], "year": [{"text": "1990s", "confidence": 0.9951321482658386, "start": 51, "end": 56}], "actor": [{"text": "bill murray", "confidence": 0.995478093624115, "start": 66, "end": 77}], "title": [{"text": "show me a comedy", "confidence": 0.9507251977920532, "start": 0, "end": 16}]}}

### NER mit_movie labels-only threshold=0.01 | show me a comedy directed by wes anderson from the 1990s starring bill murray
{"entities": {"genre": [{"text": "comedy", "confidence": 0.9873633980751038, "start": 10, "end": 16}], "director": [{"text": "wes anderson", "confidence": 0.9990099668502808, "start": 29, "end": 41}], "year": [{"text": "1990s", "confidence": 0.9951321482658386, "start": 51, "end": 56}], "actor": [{"text": "bill murray", "confidence": 0.995478093624115, "start": 66, "end": 77}], "title": [{"text": "show me a comedy", "confidence": 0.9507251977920532, "start": 0, "end": 16}]}}

### NER mit_movie with-descriptions threshold=0.5 | show me a comedy directed by wes anderson from the 1990s starring bill murray
{"entities": {"genre": [{"text": "comedy", "confidence": 0.9945589900016785, "start": 10, "end": 16}], "director": [{"text": "wes anderson", "confidence": 0.9993439316749573, "start": 29, "end": 41}], "year": [{"text": "1990s", "confidence": 0.9967100620269775, "start": 51, "end": 56}], "actor": [{"text": "bill murray", "confidence": 0.9966750144958496, "start": 66, "end": 77}], "title": [{"text": "show me a comedy", "confidence": 0.8400681614875793, "start": 0, "end": 16}]}}

### NER mit_movie threshold=0.0 candidate counts per label | show me a comedy directed by wes anderson from the 1990s starring bill murray
{"genre": 15, "director": 13, "year": 14, "actor": 14, "title": 10}

### REL threshold=0.5 | Geoffrey Hinton, a professor at the University of Toronto, won the Turing Award in 2018 together with Yann LeCun.
{"relation_extraction": {"role": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.954018771648407}, "tail": {"text": "professor", "start": 19, "end": 28, "confidence": 0.954018771648407}}], "win-defeat": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.911794900894165}, "tail": {"text": "Yann LeCun", "start": 102, "end": 112, "confidence": 0.911794900894165}}], "general-affiliation": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.9209542870521545}, "tail": {"text": "University of Toronto", "start": 36, "end": 57, "confidence": 0.9209542870521545}}], "temporal": [{"head": {"text": "Turing Award", "start": 67, "end": 79, "confidence": 0.5488344430923462}, "tail": {"text": "2018", "start": 83, "end": 87, "confidence": 0.5488344430923462}}], "physical": [], "origin": []}}

### REL threshold=0.01 | Geoffrey Hinton, a professor at the University of Toronto, won the Turing Award in 2018 together with Yann LeCun.
{"relation_extraction": {"role": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.954018771648407}, "tail": {"text": "professor", "start": 19, "end": 28, "confidence": 0.954018771648407}}], "win-defeat": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.911794900894165}, "tail": {"text": "Yann LeCun", "start": 102, "end": 112, "confidence": 0.911794900894165}}], "general-affiliation": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.9209542870521545}, "tail": {"text": "University of Toronto", "start": 36, "end": 57, "confidence": 0.9209542870521545}}], "temporal": [{"head": {"text": "Turing Award", "start": 67, "end": 79, "confidence": 0.5488344430923462}, "tail": {"text": "2018", "start": 83, "end": 87, "confidence": 0.5488344430923462}}], "physical": [], "origin": []}}

### REL threshold=0.0 | Geoffrey Hinton, a professor at the University of Toronto, won the Turing Award in 2018 together with Yann LeCun.
{"relation_extraction": {"role": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.954018771648407}, "tail": {"text": "professor", "start": 19, "end": 28, "confidence": 0.954018771648407}}], "win-defeat": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.911794900894165}, "tail": {"text": "Yann LeCun", "start": 102, "end": 112, "confidence": 0.911794900894165}}], "general-affiliation": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.9209542870521545}, "tail": {"text": "University of Toronto", "start": 36, "end": 57, "confidence": 0.9209542870521545}}], "temporal": [{"head": {"text": "Turing Award", "start": 67, "end": 79, "confidence": 0.5488344430923462}, "tail": {"text": "2018", "start": 83, "end": 87, "confidence": 0.5488344430923462}}], "physical": [], "origin": []}}

### REL threshold=0.5 | Marie Curie was born in Warsaw and later worked at the University of Paris.
{"relation_extraction": {"role": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.5605862140655518}, "tail": {"text": "University of Paris", "start": 55, "end": 74, "confidence": 0.5605862140655518}}], "general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.6475289463996887}, "tail": {"text": "University of Paris", "start": 55, "end": 74, "confidence": 0.6475289463996887}}], "origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9941626191139221}, "tail": {"text": "Warsaw", "start": 24, "end": 30, "confidence": 0.9941626191139221}}], "win-defeat": [], "physical": [], "temporal": []}}

### REL threshold=0.01 | Marie Curie was born in Warsaw and later worked at the University of Paris.
{"relation_extraction": {"role": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.5605862140655518}, "tail": {"text": "University of Paris", "start": 55, "end": 74, "confidence": 0.5605862140655518}}], "general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.6475289463996887}, "tail": {"text": "University of Paris", "start": 55, "end": 74, "confidence": 0.6475289463996887}}], "origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9941626191139221}, "tail": {"text": "Warsaw", "start": 24, "end": 30, "confidence": 0.9941626191139221}}], "win-defeat": [], "physical": [], "temporal": []}}

### REL threshold=0.0 | Marie Curie was born in Warsaw and later worked at the University of Paris.
{"relation_extraction": {"role": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.5605862140655518}, "tail": {"text": "University of Paris", "start": 55, "end": 74, "confidence": 0.5605862140655518}}], "general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.6475289463996887}, "tail": {"text": "University of Paris", "start": 55, "end": 74, "confidence": 0.6475289463996887}}], "origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9941626191139221}, "tail": {"text": "Warsaw", "start": 24, "end": 30, "confidence": 0.9941626191139221}}], "win-defeat": [], "physical": [], "temporal": []}}

### REL threshold=0.5 | Marie Curie est née à Varsovie et a ensuite travaillé à l'Université de Paris.
{"relation_extraction": {"general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.6551318168640137}, "tail": {"text": "Université de Paris", "start": 58, "end": 77, "confidence": 0.6551318168640137}}], "origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9921563863754272}, "tail": {"text": "Varsovie", "start": 22, "end": 30, "confidence": 0.9921563863754272}}], "temporal": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.5629732012748718}, "tail": {"text": "Université de Paris", "start": 58, "end": 77, "confidence": 0.5629732012748718}}], "role": [], "win-defeat": [], "physical": []}}

### REL threshold=0.01 | Marie Curie est née à Varsovie et a ensuite travaillé à l'Université de Paris.
{"relation_extraction": {"role": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.2539302110671997}, "tail": {"text": "Université de Paris", "start": 58, "end": 77, "confidence": 0.2539302110671997}}], "general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.6551318168640137}, "tail": {"text": "Université de Paris", "start": 58, "end": 77, "confidence": 0.6551318168640137}}], "origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9921563863754272}, "tail": {"text": "Varsovie", "start": 22, "end": 30, "confidence": 0.9921563863754272}}], "temporal": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.5629732012748718}, "tail": {"text": "Université de Paris", "start": 58, "end": 77, "confidence": 0.5629732012748718}}], "win-defeat": [], "physical": []}}

### REL threshold=0.0 | Marie Curie est née à Varsovie et a ensuite travaillé à l'Université de Paris.
{"relation_extraction": {"role": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.2539302110671997}, "tail": {"text": "Université de Paris", "start": 58, "end": 77, "confidence": 0.2539302110671997}}], "general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.6551318168640137}, "tail": {"text": "Université de Paris", "start": 58, "end": 77, "confidence": 0.6551318168640137}}], "origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9921563863754272}, "tail": {"text": "Varsovie", "start": 22, "end": 30, "confidence": 0.9921563863754272}}], "temporal": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.5629732012748718}, "tail": {"text": "Université de Paris", "start": 58, "end": 77, "confidence": 0.5629732012748718}}], "win-defeat": [], "physical": []}}

### REL with-descriptions threshold=0.5 | Marie Curie was born in Warsaw and later worked at the University of Paris.
{"relation_extraction": {"role": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9598538279533386}, "tail": {"text": "University of Paris", "start": 55, "end": 74, "confidence": 0.9598538279533386}}], "win-defeat": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.6923333406448364}, "tail": {"text": "University of Paris", "start": 55, "end": 74, "confidence": 0.6923333406448364}}], "origin": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9929139614105225}, "tail": {"text": "Warsaw", "start": 24, "end": 30, "confidence": 0.9929139614105225}}], "general-affiliation": [{"head": {"text": "Marie Curie", "start": 0, "end": 11, "confidence": 0.9262865781784058}, "tail": {"text": "University of Paris", "start": 55, "end": 74, "confidence": 0.9262865781784058}}]}}

### CLS multi_label cls_threshold=0.5 | Overexpression of VEGF promoted tumor angiogenesis and increased the metastatic potential of breast cancer cells.
{"hallmarks": [{"label": "inducing angiogenesis", "confidence": 0.9018980860710144}]}

### CLS multi_label cls_threshold=0.0 | Overexpression of VEGF promoted tumor angiogenesis and increased the metastatic potential of breast cancer cells.
{"hallmarks": [{"label": "sustaining proliferative signaling", "confidence": 0.04824339225888252}, {"label": "evading growth suppressors", "confidence": 0.005611483473330736}, {"label": "resisting cell death", "confidence": 0.013070600107312202}, {"label": "enabling replicative immortality", "confidence": 0.02725301869213581}, {"label": "inducing angiogenesis", "confidence": 0.9018980860710144}, {"label": "activating invasion and metastasis", "confidence": 0.3586326539516449}, {"label": "genomic instability and mutation", "confidence": 0.004136709496378899}, {"label": "tumor promoting inflammation", "confidence": 0.09780865162611008}, {"label": "deregulating cellular energetics", "confidence": 0.005752468481659889}, {"label": "avoiding immune destruction", "confidence": 0.00497996062040329}]}

### CLS multi_label cls_threshold=0.5 | Loss of p53 function allowed cells to escape apoptosis and accumulate chromosomal aberrations.
{"hallmarks": [{"label": "genomic instability and mutation", "confidence": 0.9517465829849243}]}

### CLS multi_label cls_threshold=0.0 | Loss of p53 function allowed cells to escape apoptosis and accumulate chromosomal aberrations.
{"hallmarks": [{"label": "sustaining proliferative signaling", "confidence": 0.004422913305461407}, {"label": "evading growth suppressors", "confidence": 0.0421440415084362}, {"label": "resisting cell death", "confidence": 0.4838660657405853}, {"label": "enabling replicative immortality", "confidence": 0.04507558047771454}, {"label": "inducing angiogenesis", "confidence": 0.002928712870925665}, {"label": "activating invasion and metastasis", "confidence": 0.0029325180221349}, {"label": "genomic instability and mutation", "confidence": 0.9517465829849243}, {"label": "tumor promoting inflammation", "confidence": 0.0007802228792570531}, {"label": "deregulating cellular energetics", "confidence": 0.08319707214832306}, {"label": "avoiding immune destruction", "confidence": 0.07889074087142944}]}

### CLS multi_label cls_threshold=0.5 | Patients were recruited from three hospitals between 2005 and 2010.
{"hallmarks": [{"label": "resisting cell death", "confidence": 0.037288155406713486}]}

### CLS multi_label cls_threshold=0.0 | Patients were recruited from three hospitals between 2005 and 2010.
{"hallmarks": [{"label": "sustaining proliferative signaling", "confidence": 0.014702411368489265}, {"label": "evading growth suppressors", "confidence": 0.008044914342463017}, {"label": "resisting cell death", "confidence": 0.037288155406713486}, {"label": "enabling replicative immortality", "confidence": 0.004158650524914265}, {"label": "inducing angiogenesis", "confidence": 0.01784922368824482}, {"label": "activating invasion and metastasis", "confidence": 0.013903641141951084}, {"label": "genomic instability and mutation", "confidence": 0.006184895522892475}, {"label": "tumor promoting inflammation", "confidence": 0.0015037718694657087}, {"label": "deregulating cellular energetics", "confidence": 0.0013653140049427748}, {"label": "avoiding immune destruction", "confidence": 0.01935042440891266}]}

### CLS multi_label cls_threshold=0.5 | La surexpression du VEGF a favorisé l'angiogenèse tumorale et augmenté le potentiel métastatique des cellules.
{"hallmarks": [{"label": "tumor promoting inflammation", "confidence": 0.40677809715270996}]}

### CLS multi_label cls_threshold=0.0 | La surexpression du VEGF a favorisé l'angiogenèse tumorale et augmenté le potentiel métastatique des cellules.
{"hallmarks": [{"label": "sustaining proliferative signaling", "confidence": 0.16587956249713898}, {"label": "evading growth suppressors", "confidence": 0.02614312805235386}, {"label": "resisting cell death", "confidence": 0.06038179621100426}, {"label": "enabling replicative immortality", "confidence": 0.06236463412642479}, {"label": "inducing angiogenesis", "confidence": 0.3639376759529114}, {"label": "activating invasion and metastasis", "confidence": 0.28410476446151733}, {"label": "genomic instability and mutation", "confidence": 0.021227695047855377}, {"label": "tumor promoting inflammation", "confidence": 0.40677809715270996}, {"label": "deregulating cellular energetics", "confidence": 0.03667667508125305}, {"label": "avoiding immune destruction", "confidence": 0.028568066656589508}]}

### CLS Classifier.classify (all-label probabilities) | Overexpression of VEGF promoted tumor angiogenesis and increased the metastatic potential of breast cancer cells.
{"hallmarks": {"value": ["inducing angiogenesis"], "confidence": 0.9267139521488136, "probabilities": {"sustaining proliferative signaling": 0.04824339449536442, "evading growth suppressors": 0.005611483600040049, "resisting cell death": 0.013070600240359703, "enabling replicative immortality": 0.027253017658054914, "inducing angiogenesis": 0.9018980863365007, "activating invasion and metastasis": 0.3586326639014823, "genomic instability and mutation": 0.004136709455734602, "tumor promoting inflammation": 0.09780864466795801, "deregulating cellular energetics": 0.005752468687801415, "avoiding immune destruction": 0.0049799607377239525}}, "_meta": {"feasible": true, "decoder": "independent", "exact": true, "objective": 2.2184946537017822, "violations": []}}

### CLS Classifier.classify (all-label probabilities) | Loss of p53 function allowed cells to escape apoptosis and accumulate chromosomal aberrations.
{"hallmarks": {"value": ["genomic instability and mutation"], "confidence": 0.9066354140482826, "probabilities": {"sustaining proliferative signaling": 0.004422913412901224, "evading growth suppressors": 0.042144043142795684, "resisting cell death": 0.483866025102274, "enabling replicative immortality": 0.04507558001371048, "inducing angiogenesis": 0.002928712615727903, "activating invasion and metastasis": 0.002932517801908449, "genomic instability and mutation": 0.9517465768615685, "tumor promoting inflammation": 0.0007802228757763235, "deregulating cellular energetics": 0.08319707266508947, "avoiding immune destruction": 0.07889074298297442}}, "_meta": {"feasible": true, "decoder": "independent", "exact": true, "objective": 2.9818320274353027, "violations": []}}

### CLS Classifier.classify (all-label probabilities) | Patients were recruited from three hospitals between 2005 and 2010.
{"hallmarks": {"value": [], "confidence": null, "probabilities": {"sustaining proliferative signaling": 0.01470241113883633, "evading growth suppressors": 0.008044914626862142, "resisting cell death": 0.03728815706472002, "enabling replicative immortality": 0.004158650356111254, "inducing angiogenesis": 0.017849223280318537, "activating invasion and metastasis": 0.013903640634190606, "genomic instability and mutation": 0.006184895483047209, "tumor promoting inflammation": 0.001503771860965763, "deregulating cellular energetics": 0.0013653140473334356, "avoiding immune destruction": 0.019350423688978602}}, "_meta": {"feasible": true, "decoder": "independent", "exact": true, "objective": 0.0, "violations": []}}

### CLS Classifier.classify (all-label probabilities) | La surexpression du VEGF a favorisé l'angiogenèse tumorale et augmenté le potentiel métastatique des cellules.
{"hallmarks": {"value": [], "confidence": null, "probabilities": {"sustaining proliferative signaling": 0.16587956827284261, "evading growth suppressors": 0.02614312934360427, "resisting cell death": 0.06038179225091098, "enabling replicative immortality": 0.06236464061539009, "inducing angiogenesis": 0.3639376927746783, "activating invasion and metastasis": 0.28410476820965763, "genomic instability and mutation": 0.021227697440034606, "tumor promoting inflammation": 0.4067780977287771, "deregulating cellular energetics": 0.036676675970885544, "avoiding immune destruction": 0.028568067514944725}}, "_meta": {"feasible": true, "decoder": "independent", "exact": true, "objective": 0.0, "violations": []}}

### JSON extract_json (record mode) threshold=0.5 | wake me up at 7 am tomorrow
{"slots": [{"time": [{"text": "7 am", "confidence": 0.9653666019439697, "start": 14, "end": 18}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON extract_json (record mode) threshold=0.01 | wake me up at 7 am tomorrow
{"slots": [{"time": [{"text": "wake me up at 7 am", "confidence": 0.02007371559739113, "start": 0, "end": 18}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "wake me up at 7 am tomorrow", "confidence": 0.01699102856218815, "start": 0, "end": 27}], "date": [{"text": "tomorrow", "confidence": 0.4892388582229614, "start": 19, "end": 27}], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "me up at 7 am", "confidence": 0.014339915476739407, "start": 5, "end": 18}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "me up at 7 am tomorrow", "confidence": 0.01019267551600933, "start": 5, "end": 27}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "up at 7 am", "confidence": 0.011700665578246117, "start": 8, "end": 18}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "at 7 am", "confidence": 0.07767635583877563, "start": 11, "end": 18}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "at 7 am tomorrow", "confidence": 0.010822776705026627, "start": 11, "end": 27}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "7", "confidence": 0.016576362773776054, "start": 14, "end": 15}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "7 am", "confidence": 0.9653666019439697, "start": 14, "end": 18}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "7 am tomorrow", "confidence": 0.028083914890885353, "start": 14, "end": 27}], "date": [{"text": "7 am tomorrow", "confidence": 0.013063750229775906, "start": 14, "end": 27}], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "am", "confidence": 0.02183961123228073, "start": 16, "end": 18}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON Schema.structure(mode=None) threshold=0.5 | wake me up at 7 am tomorrow
{"slots": [{"time": [{"text": "7 am", "confidence": 0.9653666019439697, "start": 14, "end": 18}], "date": [{"text": "tomorrow", "confidence": 0.903302788734436, "start": 19, "end": 27}], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON-as-entities threshold=0.5 | wake me up at 7 am tomorrow
{"entities": {"time": [{"text": "7 am", "confidence": 0.9968039989471436, "start": 14, "end": 18}], "date": [{"text": "tomorrow", "confidence": 0.9865570664405823, "start": 19, "end": 27}], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}}

### JSON extract_json (record mode) threshold=0.5 | play the latest album by taylor swift on spotify
{}

### JSON extract_json (record mode) threshold=0.01 | play the latest album by taylor swift on spotify
{}

### JSON Schema.structure(mode=None) threshold=0.5 | play the latest album by taylor swift on spotify
{"slots": [{"time": [], "date": [], "artist_name": [{"text": "taylor swift", "confidence": 0.9888991713523865, "start": 25, "end": 37}], "app_name": [{"text": "spotify", "confidence": 0.9649731516838074, "start": 41, "end": 48}], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": [{"text": "album", "confidence": 0.8014793992042542, "start": 16, "end": 21}]}]}

### JSON-as-entities threshold=0.5 | play the latest album by taylor swift on spotify
{"entities": {"time": [], "date": [], "artist_name": [{"text": "taylor swift", "confidence": 0.9964423775672913, "start": 25, "end": 37}], "app_name": [{"text": "spotify", "confidence": 0.9939508438110352, "start": 41, "end": 48}], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}}

### JSON extract_json (record mode) threshold=0.5 | what is the weather like
{}

### JSON extract_json (record mode) threshold=0.01 | what is the weather like
{}

### JSON Schema.structure(mode=None) threshold=0.5 | what is the weather like
{"slots": [{"time": [], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [{"text": "weather", "confidence": 0.5666725039482117, "start": 12, "end": 19}], "media_type": []}]}

### JSON-as-entities threshold=0.5 | what is the weather like
{"entities": {"time": [], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [{"text": "weather", "confidence": 0.7596843838691711, "start": 12, "end": 19}], "media_type": []}}

### JSON extract_json (record mode) threshold=0.5 | olly tell me a joke
{}

### JSON extract_json (record mode) threshold=0.01 | olly tell me a joke
{}

### JSON Schema.structure(mode=None) threshold=0.5 | olly tell me a joke
{"slots": [{"time": [], "date": [], "artist_name": [{"text": "olly", "confidence": 0.8091526031494141, "start": 0, "end": 4}], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON-as-entities threshold=0.5 | olly tell me a joke
{"entities": {"time": [], "date": [], "artist_name": [{"text": "olly", "confidence": 0.9504327774047852, "start": 0, "end": 4}], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}}

### JSON extract_json (record mode) threshold=0.5 | réveille-moi à 7 heures demain
{"slots": [{"time": [{"text": "7 heures", "confidence": 0.8599403500556946, "start": 15, "end": 23}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON extract_json (record mode) threshold=0.01 | réveille-moi à 7 heures demain
{"slots": [{"time": [{"text": "réveille-moi à 7 heures", "confidence": 0.01768452860414982, "start": 0, "end": 23}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "réveille-moi à 7 heures demain", "confidence": 0.024342356249690056, "start": 0, "end": 30}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "à 7 heures", "confidence": 0.03034309297800064, "start": 13, "end": 23}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "à 7 heures demain", "confidence": 0.03202198073267937, "start": 13, "end": 30}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "7", "confidence": 0.10758648067712784, "start": 15, "end": 16}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "7 heures", "confidence": 0.8599403500556946, "start": 15, "end": 23}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "7 heures demain", "confidence": 0.3219841420650482, "start": 15, "end": 30}], "date": [{"text": "7 heures demain", "confidence": 0.059921566396951675, "start": 15, "end": 30}], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "7 heures demain.", "confidence": 0.029887031763792038, "start": 15, "end": 31}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "heures", "confidence": 0.04425620660185814, "start": 17, "end": 23}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}, {"time": [{"text": "heures demain", "confidence": 0.014607692137360573, "start": 17, "end": 30}], "date": [{"text": "demain", "confidence": 0.11302080005407333, "start": 24, "end": 30}], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON Schema.structure(mode=None) threshold=0.5 | réveille-moi à 7 heures demain
{"slots": [{"time": [{"text": "7 heures", "confidence": 0.8599403500556946, "start": 15, "end": 23}], "date": [], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}]}

### JSON-as-entities threshold=0.5 | réveille-moi à 7 heures demain
{"entities": {"time": [{"text": "7 heures", "confidence": 0.965047299861908, "start": 15, "end": 23}], "date": [{"text": "demain", "confidence": 0.5159577131271362, "start": 24, "end": 30}], "artist_name": [], "app_name": [], "music_genre": [], "place_name": [], "weather_descriptor": [], "media_type": []}}

### COMBINED threshold=0.5 | Geoffrey Hinton, a professor at the University of Toronto, won the Turing Award in 2018 together with Yann LeCun.
{"event": [{"winner": [{"text": "Geoffrey Hinton", "confidence": 0.9993360638618469, "start": 0, "end": 15}], "prize": {"text": "Turing Award", "confidence": 0.995202898979187, "start": 67, "end": 79}, "year": {"text": "2018", "confidence": 0.9989765882492065, "start": 83, "end": 87}}], "entities": {"person": [{"text": "Yann LeCun", "confidence": 0.9992701411247253, "start": 102, "end": 112}, {"text": "Geoffrey Hinton", "confidence": 0.9989797472953796, "start": 0, "end": 15}], "organization": [{"text": "University of Toronto", "confidence": 0.9984487295150757, "start": 36, "end": 57}], "award": [{"text": "Turing Award", "confidence": 0.9981983304023743, "start": 67, "end": 79}]}, "topic": {"label": "politics", "confidence": 0.7726917266845703}, "relation_extraction": {"win-defeat": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.9317869544029236}, "tail": {"text": "Yann LeCun", "start": 102, "end": 112, "confidence": 0.9317869544029236}}], "role": [{"head": {"text": "Geoffrey Hinton", "start": 0, "end": 15, "confidence": 0.9725335836410522}, "tail": {"text": "professor", "start": 19, "end": 28, "confidence": 0.9725335836410522}}]}}

### LoRA target resolution
{"encoder": {"n_linear": 72, "top_level": ["encoder"]}, "encoder+span_rep+classifier+count_embed+count_pred": {"n_linear": 74, "top_level": ["classifier", "encoder"]}, "encoder+all_task_heads": {"n_linear": 131, "top_level": ["boundary_head", "classifier", "encoder", "record_decoder", "relation_scorer"]}, "all_task_heads": {"n_linear": 59, "top_level": ["boundary_head", "classifier", "record_decoder", "relation_scorer"]}, "encoder_leaf_names": ["dense", "key_proj", "query_proj", "value_proj"], "example_encoder_module": "encoder.encoder.layer.0.attention.output.dense", "params_total": 193581591, "params_by_module": {"encoder": 183763200, "classifier": 1182721, "boundary_head": 1619986, "record_decoder": 1108994, "relation_scorer": 5906690}}

### timing (RTX 3090, fp32, 5 repeats after warm-up)
{"ner_single_ms": 9.4, "rel_single_ms": 11.91, "cls10_single_ms": 7.17, "json_single_ms": 11.88, "combined_single_ms": 11.67, "ner_batch8_ms_per_sentence": 1.4, "ner_batch32_ms_per_sentence": 0.85, "cls10_batch32_ms_per_sentence": 1.51, "gpu_peak_mem_MiB": 1068}
```

## 8. Unconfirmed items

1. Whether `ClassificationConfig.calibrator` is really unused (helper agent, PR #173). I did not trace it.
2. Whether setting `model.processor.sampling_config` before training changes the training augmentation. The trainer reads `model.processor`, but I did not run it.
3. Full GPU determinism of training with a fixed seed. I did not run the same seed twice.
4. Whether the short PEFT `target_modules` list matches extra layers on reload.
5. Whether `sentencepiece` is needed in addition to `protobuf`.
6. The exact failure when a training label cannot match a word token (for example `Lithium` inside `Lithium-induced`). Validation passes; the target build may raise or skip. Not run.
7. Tutorial recipes, Decide details, and issue states come from the helper agent's web reading. I checked the Decide config and the model list myself.
8. How often our 4 datasets hit the candidate budget (192 per text). Not measured; our texts are short sentences.

## 9. Sources

- Package source: P = `.venv/lib/python3.11/site-packages/gliner2/` in this worktree, version 2.0.0. METADATA: `.venv/lib/python3.11/site-packages/gliner2-2.0.0.dist-info/METADATA`.
- GitHub: https://github.com/fastino-ai/GLiNER2 (README, `tutorial/`, releases https://github.com/fastino-ai/GLiNER2/releases/tag/v2.0.0, issues #34, #66, #148, PRs #160, #173, #176).
- Model configs: https://huggingface.co/fastino/gliner2.5-base-v1/resolve/main/config.json, https://huggingface.co/fastino/gliner2.5-multi-v1/resolve/main/config.json (identical except `model_name`), https://huggingface.co/fastino/GLiNER2.5-Decide/resolve/main/config.json.
- Model API: https://huggingface.co/api/models/fastino/gliner2.5-base-v1, `/commits/main`, `/paths-info/<rev>` for both models.
- Model skill file: https://huggingface.co/fastino/gliner2.5-base-v1/resolve/main/SKILL.md ("Tune thresholds on development data"; "Do not assume confidence scores form a normalized probability distribution").
- Paper: https://arxiv.org/abs/2507.18546 (EMNLP 2025 demos, https://aclanthology.org/2025.emnlp-demos.10/).
- Our design: `docs/research/2026-09-25-1113-full-merge-design.md` (lines 27-46).

## Changelog

- 2026-09-25 12:02 CEST - Created.
