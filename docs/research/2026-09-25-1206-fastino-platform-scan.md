---
title: Fastino platform scan - hosted services, recipes, tooling, and what it means for Active GLiNER
date: 2026-09-25 12:06 CEST
author: Claude (subagent)
type: research
status: frozen
---

# Fastino platform scan

Question: what does Fastino (makers of GLiNER2 and GLiNER2.5) offer as a hosted platform, and which parts can our open active-learning tool use?

Method: public web pages, public docs, the public OpenAPI file, the public GitHub and Hugging Face orgs. No sign-up, no login, no API calls with a key. All facts are as of 25 Sep 2026.

Words used here:

- **Hosted API** - Fastino runs the model on its servers; you send text over the internet and pay per use.
- **LoRA** - a small add-on to a model; you train only the add-on, not the whole model.
- **Active learning** - the tool picks the most useful unlabeled texts, a teacher labels them, the student retrains, and the loop repeats.

## 1. Short answer

1. Fastino sells one platform under two names: **Fastino** (agent.fastino.ai, api.fastino.ai) and **Pioneer** (pioneer.ai, api.pioneer.ai). The Pioneer name now redirects to Fastino. Sources: https://pioneer.ai/ (308 redirect to https://fastino.ai/), https://agent.pioneer.ai/llms.txt
2. The platform has a hosted inference API, a training-job API (LoRA or full), dataset upload, synthetic data generation, LLM auto-labeling, evaluations, inference feedback, and a "Fine-Tuning Agent". Source: https://agent.fastino.ai/llms-full.txt
3. The open-source library (`gliner2`, Apache-2.0) already has local LoRA training, W&B logging, and a thin API client. Source: https://github.com/fastino-ai/GLiNER2
4. Nobody (Fastino or community) ships an open active-learning loop for GLiNER2 across NER, relations, classification, and JSON. This is our gap. (Based on searches in section 5; absence is not proof.)

## 2. What the platform offers

| Piece | What it does | Status | Source |
|---|---|---|---|
| Inference, OpenAI style | `POST /v1/chat/completions` with a `schema` object (entities, classifications, structures, relations). Works for base models and fine-tuned job UUIDs. | Public, documented | https://docs.fastino.ai/inference.md |
| Inference, native GLiNER2 | `POST /v1/gliner-2` (sync), `/v1/gliner-2/async` (async jobs). Always uses `fastino/gliner2-base-v1`. Accepts a list of texts. | Public, documented | https://docs.fastino.ai/inference.md, https://docs.fastino.ai/openapi.json |
| Python client | `GLiNER2.from_api()` / `GLiNER2API` in the `gliner2` package; default base URL `https://api.fastino.ai`, key from `FASTINO_API_KEY`; calls the `gliner-2` route. | Open source | https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/7-api.md, https://github.com/fastino-ai/GLiNER2/blob/main/gliner2/api_client.py |
| Hosted model catalog | GLiNER-2.5-Decide, `fastino/gliner2.5-multi-v1`, GLiNER2 base/large/multi/multi-large, GLiGuard, PII filter. `gliner2.5-base-v1` is **not** listed (unconfirmed for the live `GET /v1/base-models` list). | Public | https://docs.fastino.ai/concepts/models.md |
| Training jobs | `POST /v1/training-jobs`; LoRA or full; SFT, GRPO, DPO; checkpoints, logs, billing, deploy, weight download, push to HF Hub. | Public, documented | https://docs.fastino.ai/training.md, https://docs.fastino.ai/openapi.json |
| Datasets | Upload JSON, JSONL, CSV (up to 50 MB) via presigned URL; types `ner`, `classification`, `custom`, `decoder`; purposes `training`, `evaluation`, `benchmark`. | Public | https://agent.fastino.ai/llms-full.txt, https://docs.fastino.ai/openapi.json |
| Synthetic data | `POST /v1/generate` builds labeled data from a task description. | Documented in llms-full.txt | https://agent.fastino.ai/llms-full.txt |
| LLM auto-labeling | `POST /v1/generate/ner/label-existing` and `/classification/label-existing`; 1 to 1000 texts per call. | Documented in llms-full.txt | https://agent.fastino.ai/llms-full.txt |
| Evaluations | F1 (primary), precision, recall, per-entity/class breakdown; compare fine-tunes with base models and LLMs. Runs are started **only** by the Fine-Tuning Agent sandbox, not by API keys. | Documented | https://agent.fastino.ai/llms-full.txt |
| Training dashboard | "Metrics tracked in real-time: F1 score, precision, recall, loss." Job fields include `progress_percent`, `current_epoch`, `metrics`, `resolved_recipe`. | Documented; UI not seen (needs login) | https://agent.fastino.ai/llms-full.txt, https://docs.fastino.ai/openapi.json |
| Experiment tracking | Experiments with a Datasets tab (field `experiment_id`); optional `wandb_api_key` on a training job. | Documented in schema | https://docs.fastino.ai/openapi.json |
| Inference feedback | `POST /v1/inferences/{id}/feedback` with `verdict` (correct/incorrect) and `corrected_output`. | Public | https://docs.fastino.ai/openapi.json |
| Continuous adaptation | Production logs, agent curates data, retrain, evaluate best checkpoint before promotion. | Described | https://agent.fastino.ai/llms-full.txt |
| Fine-Tuning Agent (Pioneer agent) | One prompt to fine-tune and deploy; launched 21 Apr 2026. | Product | https://www.prnewswire.com/news-releases/fastino-launches-pioneer-the-first-agent-for-fine-tuning-and-inference-of-llms-302748105.html |
| Pioneer agent paper | arXiv 2604.09791: cold-start and production modes; diagnosis, curriculum synthesis, retraining, verification; AdaptFT-Bench. Entity F1 0.345 to 0.810 in a production case. | Paper | https://arxiv.org/abs/2604.09791 |
| GLiNER2.5-Decide | 340M typed-decision model, Apache-2.0, `fastino/GLiNER2.5-Decide` on HF; hosted as `fastino/gliner2.5-decide`. 60.1% avg on their 17-dataset Fast Decisions benchmark. | Released 24 Sep 2026 | https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model, https://docs.fastino.ai/concepts/models.md |
| Agent skill | Hosted GLiNER skill for coding agents. | Documented | https://docs.fastino.ai/concepts/gliner-agent-skill.md |
| Decoder LLMs | LoRA training of Qwen3, Llama 3.x, Gemma 4, gpt-oss, DeepSeek V3.1; serverless inference for some. | Documented | https://agent.fastino.ai/llms-full.txt |

## 3. Pricing, limits, terms

| Item | Fact | Source |
|---|---|---|
| Plans | Pro: $20/seat/month, includes $40 platform credits, weight download. Enterprise: $50/seat/month, includes $50 credits, SSO, "Inference-tracking opt-out". | https://pioneer.ai/pricing |
| Free tier | FAQ says "Free to experiment". Size of the free tier: unconfirmed. An older 2025 report says 10,000 requests/month free; this may be out of date. | https://agent.fastino.ai/llms-full.txt, https://dataphoenix.info/fastino-secures-17-5m-for-its-task-specific-models-trained-on-low-end-gpus/ |
| Per-token price | $0.90 / $0.90 per 1M tokens for GLiNER2 Large: seen only in a search snippet; unconfirmed. | https://fastino.ai/models/gliner2 (snippet, not on fetched page) |
| Training cost | Billed by GPU minutes (`gpu_minutes`, `charged_usd`). Rate per minute: unconfirmed. Agent runs average about 6 hours and about $35. | https://docs.fastino.ai/openapi.json, https://www.prnewswire.com/news-releases/fastino-launches-pioneer-the-first-agent-for-fine-tuning-and-inference-of-llms-302748105.html |
| Academic discount | Special pricing for students, non-profits, and open source via an intake form. | https://agent.fastino.ai/llms-full.txt |
| Rate limits | Numbers not published. API returns `429` with `Retry-After`; `402` for spend limits; cold starts need a 300 s timeout. | https://docs.fastino.ai/inference.md |
| Storage | Dataset storage is free. | https://agent.fastino.ai/llms-full.txt |
| Training on your data | Default yes: "Unless you opt out, we may use inputs and outputs you send through the API to improve and train our models." Opt-out for Enterprise (trust page) or Pro and Custom (FAQ); the two pages disagree. | https://docs.fastino.ai/trust-safety.md, https://agent.fastino.ai/llms-full.txt |
| Retention | Inputs and outputs kept indefinitely by default. `store: false` gives zero retention "for eligible use cases". | https://docs.fastino.ai/trust-safety.md |
| Subprocessors | 15 listed, including AWS, Anthropic, OpenAI, Modal, Azure (US). | https://docs.fastino.ai/trust-safety.md |
| GDPR | "At this time, we do not offer a Data Processing Addendum (DPA)." | https://docs.fastino.ai/trust-safety.md |
| Compliance | SOC 2 Type II and ISO 27001 in progress; first audit expected Nov 2026. | https://docs.fastino.ai/trust-safety.md |
| Terms of Use | The terms page is a JavaScript app; the text did not load without a browser. Clauses on output ownership, benchmarking, or publication: unconfirmed. | https://agent.pioneer.ai/terms |
| Open weights | All GLiNER2, GLiNER2.5, and Decide models on HF are Apache-2.0. Local results have no platform terms. | https://fastino.ai/models/gliner2-5, https://huggingface.co/fastino/gliner2-base-v1 |

For a paper: results from open weights run locally are safe to publish. Results from the hosted API depend on terms we could not read; treat them as unconfirmed until someone reads the terms in a browser.

## 4. Recipes, formats, and public material

### 4.1 LoRA and training defaults

| Setting | Local library (`TrainingConfig`) | Hosted training API |
|---|---|---|
| LoRA rank `r` | 16 default; tutorial shows 8 (less memory) and 32 (more capacity) | 16 default (non-decoder) |
| LoRA alpha | 32 ("typically 2*r") | 32 default |
| LoRA dropout | 0.0 default; tutorial suggests 0.1 | 0.1 default |
| Target modules | `encoder`, `span_rep`, `classifier`, `count_embed`, `count_pred` (all by default) | not exposed |
| Learning rate | `encoder_lr=1e-5`, `task_lr=5e-4`; with LoRA, `task_lr` drives adapter and heads | `learning_rate` 2e-5 default; separate `encoder_learning_rate`, `task_learning_rate` |
| Schedule | linear, `warmup_ratio=0.1` | cosine default |
| Batch | 32 | 4 (catalog default may apply) |
| Epochs | 10 | up to 100 with early stopping (patience 3) |
| Validation | `eval_strategy` steps/epoch, `metric_for_best="eval_loss"` | 20% held out by default |
| Seed | 42 | 3407 default; only honored when pinned to provider `modal` |
| Data sizing | none | optional `auto_data_sizing`: min(max, max(min, samples_per_label x num_labels)) |

Sources: https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md, https://docs.fastino.ai/openapi.json (schema `TrainingJobCreate`).

Note: both defaults pick the best checkpoint by **loss**, not F1. A community thread says F1 "does not correlate well with loss after some point". Source: https://github.com/urchade/GLiNER/discussions/87

### 4.2 Data formats

| Format | Shape | Source |
|---|---|---|
| Local `gliner2` JSONL | `{"input": ..., "output": {...}}` or `{"text": ..., "schema": {...}}`; keys `entities`, `entity_descriptions`, `classifications` (with `true_label`, `multi_label`), `json_structures`, `json_descriptions`, `relations`. Same format for GLiNER2.5. | https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/8-train_data.md |
| Hosted NER upload | `{"text": ..., "entities": [["Apple", "ORG"], ...]}` | https://agent.fastino.ai/llms-full.txt |
| Hosted classification | `{"text", "label"}` or `{"text", "labels"}` | https://agent.fastino.ai/llms-full.txt |
| Hosted inference schema | `entities` list of `{name, description}`; `structures` as `field::type::description`; `relations` as names | https://docs.fastino.ai/inference.md |

The hosted upload format covers NER and classification only in the docs. Relations and JSON upload formats are not documented there (unconfirmed; "JSON Extraction" is described as "Same schema as NER").

### 4.3 GitHub org `fastino-ai` (7 public repos, checked via GitHub API)

| Repo | Stars | Last push | What it is |
|---|---|---|---|
| GLiNER2 | 2176 | 2026-09-24 | Main library: inference, training, LoRA, API client, 16 tutorials |
| GLiGuard | 60 | 2026-05-12 | LLM guardrail model |
| pioneer-example | 14 | 2025-11-18 | Example of Pioneer for agent personalization |
| PROBE_benchmark | 13 | 2025-10-29 | Code for "Beyond Reactivity" proactive-agent benchmark |
| preconditioner-bias-correction | 4 | 2026-05-19 | No description |
| mintlify-docs | 2 | 2026-09-24 | Source of docs.fastino.ai |
| vllm-factory-fstn | 1 | 2026-09-24 | vLLM plugins for encoder models (ColBERT, GLiNER, embeddings) |

Source: https://github.com/fastino-ai

GLiNER2 tutorials (1 to 16) cover classification, NER, JSON, combined, validators, relations, API, training data, training, LoRA adapters, adapter switching, long context, span attributes, constrained classification, joint IE, and PII. Source: https://github.com/fastino-ai/GLiNER2/tree/main/tutorial

### 4.4 Hugging Face org `fastino`

| Kind | Items | Source |
|---|---|---|
| Models (14) | gliner2-base-v1, gliner2-large-v1, gliner2-multi-v1, gliner2.5-base-v1, gliner2.5-multi-v1, gliner2.5-small-v1, GLiNER2.5-Decide, GLiNER2.5-multi-Decide, GLiNER2.5-Decide-1B, gliner2-privacy-filter-PII-multi, GLiNER2-Guardrails-PII-Multi, gliguard-LLMGuardrails-300M, Fastino-Nemotron-3.5-Lightning-Finance, -Healthcare | https://huggingface.co/fastino |
| Datasets (1) | `fastino/fast-decisions` (Apache-2.0, text classification, 1K to 10K rows) | https://huggingface.co/datasets/fastino/fast-decisions |
| Spaces (8) | gliner2-official-demo, gliner25-decide-playground, GLiGuard, gliner2-guardrails-pii-multi, gliner25-span-attributes, gliner25-long-context, gliner25-constrained-classification, gliner25-unlimited-span-length | https://huggingface.co/fastino |

### 4.5 Published benchmark numbers

| Model | Number | Source |
|---|---|---|
| GLiNER2.5 base | Few-NERD F1 55.14; XNLI 54.49 | https://fastino.ai/models/gliner2-5 |
| GLiNER2.5 multi | Few-NERD F1 52.37; XNLI 62.30 | https://fastino.ai/models/gliner2-5 |
| GLiNER2.5-Decide | 60.1% avg over 17 datasets (5,100 examples, their own benchmark) | https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model |
| GLiNER2 (paper) | Zero-shot NER and classification tables in the PDF; not extracted here | https://arxiv.org/abs/2507.18546 |

## 5. Open-source tooling (Fastino and community)

| Need | What exists | Source |
|---|---|---|
| Training curve logging | `gliner2` trainer: W&B only (`report_to_wandb`, project, entity, tags). No TensorBoard or MLflow found in the repo. | https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md |
| Hosted W&B | `wandb_api_key` field on hosted training jobs. | https://docs.fastino.ai/openapi.json |
| Metrics code | `gliner2/training/metrics.py`: `f1_from_counts`, `exact_span_counts`, `boundary_recall`, `candidate_oracle_recall`. Custom `compute_metrics` hook in trainer. | https://github.com/fastino-ai/GLiNER2/tree/main/gliner2/training |
| Error analysis, confusion views | None found in the open library. The hosted eval has per-class breakdown. | https://agent.fastino.ai/llms-full.txt |
| Label Studio | Official GLiNER ML backend (pre-labeling, interactive labeling) for the original GLiNER. GLiNER2 support: unconfirmed. | https://labelstud.io/guide/ml_tutorials/gliner, https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/gliner |
| Argilla | Tutorial uses GLiNER (gliner_mediumv2) for token-classification suggestions. | https://docs.argilla.io/dev/tutorials/token_classification/ |
| Synthetic data + fine-tune | `gliner-finetune` (GPT synthetic data, original GLiNER; last push June 2024). | https://github.com/wjbmattingly/gliner-finetune |
| Scripted fine-tune | `train-gliner2.py` / `classify-gliner2.py` uv scripts (PR closed 23 Sep 2026). | https://github.com/davanstrien/uv-scripts-for-ai/pull/118 |
| Decide eval and LoRA | `decide-lab`: independent eval, calibration, LoRA of GLiNER2.5-Decide (0 stars, 24 Sep 2026). | https://github.com/turlockmike/decide-lab |
| Original GLiNER fine-tune | Community script with grad accumulation and mixed precision. | https://github.com/urchade/GLiNER/discussions/87 |
| Active learning | No open GLiNER or GLiNER2 active-learning loop found. Closest is the closed Pioneer agent (diagnose failures, synthesize curriculum, retrain). | https://arxiv.org/abs/2604.09791 |

## 6. Recommendation for Active GLiNER

### 6.1 Use or integrate

| Piece | How we use it | Why |
|---|---|---|
| `gliner2` library + open weights | Core student, local LoRA training on the 3090 | Apache-2.0, reproducible, no account |
| `gliner2` JSONL format | Our canonical data format for all 4 tasks | It already covers entities, relations, classifications (multi-label), and JSON structures in one file |
| LoRA defaults (r=16, alpha=32, all module groups, `task_lr=5e-4`) | Starting point and "vendor default" baseline row | Lets readers match our setup to the official recipe |
| Published numbers (Few-NERD, XNLI, Decide) | Cited as reference points only | Different splits and settings; not a direct baseline |
| W&B hook in the trainer | Optional logger | Already built in; we add a local logger as default |
| Hosted API (optional backend) | Plug-in "remote student" or "remote teacher" behind a flag; off by default | Useful for users without a GPU; never needed for paper numbers |
| Feedback schema (`verdict`, `corrected_output`) | Idea for our correction record format | Matches the shape of a human-in-the-loop correction |

### 6.2 Do not depend on

| Piece | Reason |
|---|---|
| Hosted training and evals | Closed code; eval runs only through the agent sandbox; seed honored only on one provider; cost per GPU minute unpublished |
| Hosted inference for paper results | Default retention is indefinite and data may train their models; no DPA; terms text unread; model versions can change silently |
| Hosted `/v1/gliner-2` route | Fixed to `gliner2-base-v1`, not our GLiNER2.5-base student |
| Hosted catalog for GLiNER2.5-base | Not listed in the docs catalog (unconfirmed for the live list) |
| Synthetic data and label-existing endpoints as teachers | Teacher model and prompts are not disclosed, so the paper cannot state what labeled the data |
| Pioneer agent results as a baseline | No public code; AdaptFT-Bench release status unconfirmed |

### 6.3 What our open tool offers that the platform does not

1. **A real active-learning loop.** Uncertainty and diversity sampling for all 4 tasks, with budget curves. The platform retrains on logs and feedback but documents no query strategy.
2. **Open, named teachers.** Any LLM (local or API), with prompts and costs logged, so each label has a known source.
3. **Full reproducibility.** Fixed seeds, pinned model hashes, local 3090, config files in git.
4. **F1-based checkpoint choice.** Both official defaults choose by loss; we choose by task F1.
5. **Error analysis the library lacks.** Per-label confusion, boundary errors, relation direction errors, JSON field-level errors, teacher-vs-student disagreement.
6. **Local logging.** CSV/JSONL curves and TensorBoard or MLflow as options, not only W&B.
7. **Annotation bridges.** Export to Label Studio and Argilla for human review of the chosen samples.
8. **No data leaves the machine** unless the user turns on a remote backend.

## 7. Open questions

1. Terms of Use text: output ownership, benchmark publication, competitive use. Read https://agent.pioneer.ai/terms in a browser.
2. Free tier size and per-token and per-GPU-minute prices. Check https://pioneer.ai/pricing after login, or ask support@fastino.ai.
3. Does the live `GET /v1/base-models` list `fastino/gliner2.5-base-v1` for inference or training?
4. Is AdaptFT-Bench public?

## Changelog

- 2026-09-25 12:06 CEST - Created.
