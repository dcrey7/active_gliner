---
title: Active GLiNER paper - full merge design (NER + relations + classification + slot JSON)
date: 2026-09-25 11:13 CEST
author: Claude (draft), Codex gpt-6-astra (review)
type: research
status: final - signed off by Codex (round 12)
---

# Active GLiNER paper - full merge design

Abhishek's decision (25 Sep, just before 11:13): one paper with all four tasks, not NER first and multi-task later. With Codex writing code and the 3090 running around the clock, the limit is GPU time, not his hours.

This design extends `2026-09-25-1040-paper-direction-decision.md` (signed off, round 6). Every rule there stays unless this file changes it: pool rules (3b), pins and timing gate (3d-bis), analysis rules (3e), cost accounting (3f), tool and release (4), hero figures, title gates (4b), thesis fixes (5).

## 1. One question, four output structures

+ Framing (Codex round 8): "When does student uncertainty improve one-round LLM annotation, and when does teacher error erase that benefit?"
+ The four tasks test that same interaction across four structured prediction outputs: spans (NER), typed pairs (relations), label sets (multi-label classification), slot records (JSON).
+ Say "structured prediction", not "extraction", because classification is not extraction.
+ Never average unlike metrics into one headline number. Report per task.
+ Joint selection (section 3) is exploratory.
+ Pilot evidence from the thesis (MIT Movie, one seed): on cumulative subsets of the training pool where zero-shot GLiNER was least confident, Gemma 3 12B scored 56.9 F1 (worst 100), 60.7 (worst 500), 64.3 (worst 1,000), 68.2 (worst 2,500) (`results2/exp_confidence_gliner_llm_baseline_f1`); on the full test set it scored 69.4 (`results2/exp_gliner_llm_threshold_f1`). This suggests shared difficulty, but the pilot has no matched within-pool comparison and the curve is not strictly monotonic. Students trained on Gemma labels of those sentences reached about 69-70 F1 on the test set (69.05 at N = 2,500, peak 69.87); ground truth on the same sentences reached 85.21 at N = 2,500 (`results2/exp_confidence_gliner_llm_ft_f1`). Four teachers were tried (Mistral 7B about 64, Gemma 3 12B about 69, Qwen3 235B instruct about 75, reasoning about 77 F1; thesis section 5.1); their error similarity was described only qualitatively. This paper tests the pilot properly: disjoint confidence bins on matched sentences, identical sentences for all teachers, 6 datasets, seeds.

## 2. Tasks, data, confidence

| Task | Dataset (ground truth) | Pool | Test | Sentence uncertainty (min rule) | Primary metric |
|---|---|---|---|---|---|
| NER | CleanCoNLL (CoNLL-03 text + corrected labels, built with the official script), BC5CDR (`tner/bc5cdr`, expert biomedical), MIT Movie fixed (`rungalileo/mit_movies_fixed_connll_format`, thesis link) | full train (each under 15k) | official test | min entity confidence; no prediction = 0 | strict entity micro-F1 |
| Relations | CrossRE, 6 domains pooled (`DFKI-SLT/cross_re`, expert, GPL-3.0) | train + dev (about 2.8k) minus a seeded 300 early-stopping set | official test (about 2.4k) | min relation confidence over predicted typed pairs; no prediction = 0 | strict directed triple micro-F1 |
| Multi-label classification | Hallmarks of Cancer (`bigbio/hallmarks_of_cancer`, one domain expert, 10 labels, GPL-3.0) | full train (about 12.1k sentences) | official test | min over all 10 labels of abs(p - t) | micro-F1 (macro-F1 secondary) |
| Slot JSON | MASSIVE 1.1 en-US (CC BY 4.0, text-aligned slots) | full train (about 11.5k) | official test | min over predicted slot values; empty-output rule below | canonical (slot, value) micro-F1; whole-record accuracy secondary |

Per-task rules:

+ Dataset choice (25 Sep, Abhishek: high quality, small, max about 15k train) - replaced CoNLL-2003 (about 7% label errors, fixed by CleanCoNLL), WNUT-17 (crowd, noisy by design), CoNLL04 (no IAA, 288 test sentences), GoEmotions (Surge AI audit: about 30% of 1,000 sampled comments mislabelled). MIT Movie kept only in the Galileo version for the thesis link; Galileo corrected about 4% of labels, it did not re-annotate the whole set. Split sizes marked unverified in the dataset scan are measured at load time and written to the pins file.
+ Licences - CleanCoNLL labels released only as sentence ids + offsets (Reuters text). CrossRE and Hallmarks of Cancer derived labels are released under GPL-3.0, separate from the Apache-2.0 code. BC5CDR licence to be confirmed before release.
+ LLM memorisation - CleanCoNLL fixes labels, not text exposure: CoNLL-03 text is widely memorised, and the CrossRE news domain also uses CoNLL-03 text. BC5CDR and Hallmarks add domain diversity; they are not proof against memorisation. The overlap audit covers all sets, including CoNLL and CrossNER ancestry. Pooled CrossRE results are mixed-domain, not unseen-domain generalisation.
+ Documents - where sentences come from one document (BC5CDR, Hallmarks abstracts), pool ids and any split are made by document, so sentences of one abstract never cross splits.
+ Dev holdouts - frozen before any prompt tuning, threshold choice or training, and used for every development decision. MIT Movie has no dev split: a seeded, document-free 10% of train is reserved as dev before "full train" is defined. CrossRE: a seeded 300 sentences from train + dev are the holdout; dev examples inspected earlier never enter the acquisition pool. This changes CrossRE's published protocol; say so.
+ Document provenance - acquisition budgets count sentences, but holdouts and audits group by source document. Hallmarks: group by PMID (strip the sentence suffix from `document_id`). BC5CDR: the TNER copy has no document ids, so load BigBio `bc5cdr` (which has PMIDs) or rebuild provenance before the audit.
+ Quality claims - Hallmarks: expert-annotated by one domain expert; cite the later CHAT second-expert check (4,963 sentences) only if it covers the release used; state annotation uncertainty. MIT Movie: Galileo corrections, about 4% of labels.
+ Licence gates before release - BC5CDR and the Galileo MIT Movie copy. If redistribution of derived labels is not allowed, release only code and configs for that set.
+ NER - as in the direction doc.
+ Relations - CrossRE multi-label pairs are kept: a directed pair can carry several relation types, and each (head, tail, type) is a separate triple. Sentences with no relation stay in the pool. Frozen token-to-character mapping, duplicate handling, and whether entity types count toward a correct triple. If GLiNER2.5 gives one pair score, use it once; if head and tail differ, use min(head, tail) and call it a proxy for relation confidence. Report predicted-pair counts and empty-output rates. Confirm whether every split has at least one relation per sentence; if so, state that the results hold only for relation-bearing sentences.
+ Classification - score all 10 labels, negatives included. The Hallmarks "none" label maps to an empty label set. Threshold t frozen on dev. `min abs(p - t)` is a nearest-boundary heuristic, not full multi-label uncertainty; say so. Sensitivity check if affordable: mean binary entropy over the 10 labels.
+ Slot JSON - E2E is dropped: its records often do not match the text (omissions and additions in 40% of an audit by its authors). MASSIVE slots are marked in the text, so records match. Output = `{slot_type: [values]}`. Score canonical (slot, value) pairs; absent-absent matches do not count. Empty-output rule: every sentence gets one confidence c on one scale, and selection picks ascending c. Non-empty output: c = min over predicted slot values (all at or above the 0.5 threshold). Empty output: c = 1 - max_candidate_probability, where max_candidate_probability is the highest slot score below the threshold (confident absence gives high c, near-threshold absence gives c near 0.5). No candidate at all: max_candidate_probability = 0, so c = 1 (confident absence). This convention is fixed before runs. Ties broken by seed. Slot values are validated as exact substrings of the utterance during loading; normalization and duplicate-value handling are frozen. Gold intents are never given to any model or used to restrict the schema. Scores are not comparable with published MASSIVE results. The overlap audit includes MASSIVE's source, SLURP. The HF MASSIVE copy uses a loading script, so load the official release files. All utterances stay in the pool, including those with no gold slot (filtering on gold slots would use hidden labels).
+ Pools - frozen pool IDs shared by all arms. Audit duplicates. Report rare-label coverage for Hallmarks labels and slot types. Call full-pool references "full eligible pool".
+ Teacher - one frozen prompt per task, fixed on dev, JSON output validated. Raw text in, no gold hints.

## 3. Joint selection (exploratory)

+ Corpus: CrossRE (entities and relations on the same sentences, same pool and test as the relation task). One joint student, NER + relations, frozen joint loss weights. Every selected sentence gets both label types.
+ Selectors: random; NER-only uncertainty; relation-only uncertainty; `min-of-tasks`; `mean-of-tasks`.
+ Combination: per-task percentile ranks with ascending midranks (ties kept), then one seeded tie-break on the combined score. Ranks remove scale differences, not calibration differences; say so.

## 4. Seeds and statistics

+ Primary cells (min vs random x GT vs Gemma at the larger N, every task and NER dataset) get 5 seeds. Everything else 3 seeds.
+ Four primary interaction contrasts, one per task, Holm-corrected together.
+ Four practical claims `(min - random)_Gemma` set the tool defaults. They are Holm-corrected as a separate family.
+ Report paired seed differences and seed spread; test-set bootstrap intervals are reported as test uncertainty only.
+ If intervals stay wide, the paper is framed as estimation (effect sizes with intervals), not as significance claims.

## 5. Run matrix

| Block | Count |
|---|---:|
| NER core: 3 selectors x {GT, Gemma} x N {100, 400} x 3 seeds x 3 datasets | 108 |
| NER primary cells, 2 extra seeds: {min, random} x {GT, Gemma} x N 400 x 3 datasets x 2 | 24 |
| NER routing: {routed, random 25%} x Gemma x N 400 x 3 seeds x 3 datasets | 18 |
| NER Qwen (CleanCoNLL only): {min, random} x N {100, 400} x 3 seeds | 12 |
| NER no-prediction sensitivity (CleanCoNLL): min x {GT, Gemma} x N 400 x 3 seeds | 6 |
| Relations, classification, slot JSON: {random, min} x {GT, Gemma} x 2 budgets x 3 seeds, x 3 tasks | 72 |
| Their primary cells, 2 extra seeds: 4 cells x 2 x 3 tasks | 24 |
| References, 1 seed each, reported as reference points: {full GT, full eligible pool Gemma} x (3 NER datasets + 3 other tasks) | 12 |
| Joint CrossRE: 5 selectors x {GT, Gemma} x N 200 x 3 seeds | 30 |
| **Total** | **306** |

Budgets: NER {100, 400}; relations {100, 200}; classification {100, 400}; slot JSON {100, 400}.

Dropped from the signed-off NER plan to pay for seeds: the MSE block (12) and Qwen on a second NER dataset (12); Qwen now runs on CleanCoNLL only. In the hero figures, the Qwen matched-coverage panel is CleanCoNLL only. Reference runs cut from 3 seeds to 1 (they are context, not tested).

Cut order if the timing gate fails: (1) the whole joint block (30), never only its single-task control arms, (2) NER Qwen, (3) N = 100 on the three non-NER tasks, (4) NER diversity selector, (5) routing. Never cut the 5-seed primary cells.

## 6. Compute budget

+ Codex round-8 planning estimate (not measured): about 240-450 GPU hours with a 20% retry margin. That is 10-19 days of the 3090 running nonstop.
+ Scope (Codex round 18): that estimate covers only the round-8 matrix (306 runs). Added since: 15 teacher-ladder runs and the ladder's teacher labelling (section 11), 6 mixing runs and 80 short tuning trials (section 12). None of these are estimated yet. Section 13 adds the multi student, 12 French runs, French labelling and the zero-shot reference ladder (56 evaluations). The day-3 timing gate measures one of each with the multi student and gives the full total for 339 main runs + 80 trials + ladder labelling + French labelling + reference inference; the cut order (section 5 plus section 13 additions) applies to that total.
+ Teacher labels: about 55k items with Gemma (NER pools about 29k: CleanCoNLL 14k, MIT Movie about 10k, BC5CDR about 5k; CrossRE about 2.5k; Hallmarks about 12.1k; MASSIVE about 11.5k), plus dev and test calls. Sizes confirmed at load time.
+ Teacher and student share one GPU: label first, then train.
+ The day-3 timing gate measures one run of every kind, including one reference run and one full test evaluation per task. The schedule and cut order are applied on those numbers.

## 7. Tool and title

+ `active-gliner run --task {ner,relations,classify,slots}`. Each task is a plug-in with its scorer, prompt, validator and metric. Defaults follow the per-task practical results.
+ Title version A: "Active GLiNER: Which Examples Should LLMs Label?". B and C (direction doc 4b) need their gate on NER and on at least 2 of the other 3 tasks; the abstract lists where it fails.

## 8. Timeline (about 3-4 weeks wall time)

```
days 1-4    Codex builds loaders, 4 task plug-ins, teacher cache, train, eval; Claude reviews.
            Pins file. Overlap and duplicate audit.
day 3-4     timing gate: one run of every kind. Apply the cut order.
days 4-7    teacher labelling (about 50k items).
days 6-20   runs 24/7.
days 15-24  analysis, charts, draft. Abhishek reads and edits (about 1 h/day).
after       endorser, arXiv, minimum launch.
```

Two weeks is fragile (Codex). Three to four is realistic.

## 9. Review log

+ Round 7 (Codex, small-merge option) - accepted with changes. Superseded by the full merge; its relation rules are in section 2.
+ Round 8 (Codex) - one paper is coherent under the framing in section 1. Required: E2E replaced (records do not match text), scorer repairs, frozen joint controls with single-task arms, separate correction for practical claims, more seeds on primary cells, a measured full budget. Estimated 240-450 GPU hours. Applied in this version.
+ Round 9 (Codex) - round-8 items met; MASSIVE confirmed suitable (validate substrings, no gold intents, audit SLURP overlap); 306 runs confirmed. Required: full empty-slot scoring rule, and never cut the joint single-task controls alone. Applied at 11:18.
+ Round 10 (Codex) - SIGN OFF. Dataset choices reopened at Abhishek's request (small, high-quality); a dataset swap goes back to Codex.
+ Round 11 (Codex) - datasets accepted; BC5CDR kept. Required: frozen dev exclusions, document provenance (PMID), CrossRE multi-label pairs kept, qualified quality and exposure claims, licence gates, Qwen wording. Applied at 11:26.
+ Round 12 (Codex) - SIGN OFF on the dataset swap.

## 10. Guideline additions (Codex round 13, signed off)

+ ASO over seeds is secondary and exploratory. Report epsilon_min, direction, confidence level and bootstrap settings. epsilon_min is a violation-ratio bound, not a p-value. It never overrides the two Holm families, never sets defaults, never rescues a failed claim.
+ Annotation volume per selector, budget and seed: input tokens under one frozen tokenizer, and gold and teacher item counts (entities; directed typed triples; positive labels; canonical slot-value pairs). Volume is not effort; measured costs stay. Gold counts are diagnostics only, never acquisition inputs.
+ Discussion of successor-model mismatch: selection uses the frozen zero-shot model, training produces the LoRA student. Results hold for this pair only.
+ Each replicate varies acquisition randomness, data order and LoRA initialisation, with separate recorded random streams matched across paired arms. Pools, holdouts, prompts and teacher caches stay fixed; selected ids are shared across gold and teacher arms.

## 11. Teacher questions (Q2-Q5), secondary and exploratory (Codex round 14)

Plain questions: Q2 do LLM label errors fall where GLiNER is uncertain or wrong? Q3 do different LLMs fail on the same sentences? Q4 do larger teachers in a family label better? Q5 do better teachers give better students?

A. Teacher ladder, labelling only
+ Teachers (newest available, 25 Sep 2026; Gemma 4 31B dropped at Abhishek's request): Qwen 3.8 2B (local, 4-bit), Gemma 4 12B (local, 4-bit, main teacher), Qwen 3.8 27B (Cerebras `qwen-3.8-27b`, full precision), gpt-oss-120b (Cerebras). Within-family pair for Q4: Qwen 3.8 2B vs 27B (precision also differs: 4-bit local vs full-precision API; stated as a confound). API outputs are cached and released, because hosted models can be deprecated (Cerebras retired Qwen3-235B in May 2026). Exact checkpoints, precision, decoding, token budgets and validators pinned. One frozen prompt per task for all teachers. Cerebras prices for both models checked in the account at O6 before labelling.
+ Sample: identical ids for every teacher; about 1,000 sentences per dataset from the official test set, stratified and document-grouped (all sentences if fewer). Sampling probabilities recorded; aggregates weighted for the stratification; bootstrap by document.
+ Freeze: prompts, teachers, selectors, training and analyses are fixed before any ladder result is inspected. Ladder results never feed back into method choices, teacher choice for B, or defaults.
+ Definitions fixed in advance: sentence correctness per task (all items exactly right); invalid or unparseable output counts as wrong and is reported separately.
+ Q2: teacher error rate against GLiNER zero-shot confidence and against GLiNER's actual errors (including empty predictions), adjusted for sentence length and gold item count.
+ Q3: per teacher marginal error rate; pairwise Jaccard of wrong sentences and Cohen's kappa on correctness, each next to the overlap expected under independence; a "hard for all" set with examples.
+ Q4 wording: "the larger checkpoint in this pair labels better or worse under this protocol". No claim that parameter count causes quality. Each task reported separately; cross-family comparisons descriptive only.
+ Cost: about 30k labelling calls; teacher time measured in the O7 timing gate.

B. Students from each teacher (random selection only)
+ {random} x N 400 x 3 seeds x {CleanCoNLL, Hallmarks} for Qwen 2B and gpt-oss-120b = 12 runs, plus Qwen 27B on Hallmarks = 3 runs. Total 15. Gemma 12B cells already exist.
+ Selected sentences, training settings and paired seeds shared across teachers. Teacher quality also measured on those training sentences.
+ A min-selection version (18 more runs) only if pre-declared: not added in this design.

Inference: A and B sit outside both Holm families. Effects with intervals. Any significance claim here gets its own pre-declared correction within this block. Nothing here changes defaults or rescues a primary claim.

Run count: 306 + 15 = 321. Cut order: block B is cut before everything in section 5's cut list; block A (no training) is cut only if teacher time fails the gate, starting with Qwen 3.8 2B.

## 12. Building on every thesis experiment (Codex round 17)

Inventory: `docs/wiki/thesis-inventory.md`.

+ Tuning (thesis E3, E4; the thesis tuned on the test set). One Optuna search per task on the frozen dev set for GLiNER2.5-base: 20 trials, median pruning, short step budget, training data = a fixed random ground-truth sample from the pool of size equal to the task's largest budget (N 400; relations N 200), objective = dev micro-F1 of the task. NER: one search on CleanCoNLL; the frozen recipe is reused unchanged for BC5CDR and MIT Movie (stated as a limitation). The tuning sample's ground-truth labels are an extra annotation cost outside the acquisition budgets (400 per task, 200 for relations); every arm shares the recipe, so the cost is reported once, separately, and disclosed in the paper. LoRA targets limited to modules that exist in GLiNER2.5 (checked at O5); layer groups are part of the search space. Dev labels are used for evaluation only, never for training. 80 auxiliary trials; their compute and dev labels are charged to the timing gate. The recipe is frozen before any main run.
+ Calibration (thesis E1, E5). Continuous bins [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0] with count, mean confidence and empirical correctness per bin. Extraction tasks (entities, directed typed triples, slot-value pairs): bins hold emitted predictions only (score at or above the 0.5 threshold), correctness = exact match with ground truth; the empty low bins are a threshold effect and are stated as such. Classification: bins hold all 10 labels per sentence, confidence = positive probability p, correctness = binary truth (a reliability diagram, negatives included). Before and after LoRA. Descriptive only.
+ Thresholds (thesis E6). Acquisition threshold (0.5, fixed) is separate from the evaluation operating threshold, which is chosen on dev and frozen. A test sweep over 0.1, 0.3, 0.5, 0.7, 0.9 is descriptive. No training.
+ Mixing (thesis E10). Adapted replication on MIT Movie (Galileo): min selection, N 400, Gemma for the rest, matched ids and seeds; 25% reuses the existing random-25% routing cell; 50% and 75% are new: {50, 75} x 3 seeds = 6 runs. Secondary.
+ Prior-work section names the thesis problems: tuning on the test set, the pool-filter leak, LLM F1 values in the cost table that were assumed, no reported seed replication.

Run count: 321 + 6 = 327.

## 13. Student model and a second language (Abhishek, 25 Sep 12:11)

+ Student - `fastino/gliner2.5-multi-v1` (287M, mDeBERTa-v3-base, Apache-2.0, revision pinned at O3), replacing GLiNER2.5-base everywhere in this design. Abhishek's decision: multi is the strongest general checkpoint; real use cases (for example in France) are multilingual, and the expected English gap to base is small (about 1-2 F1, to be measured by the reference ladder). No GLiNER2.5 large exists (Hub check 25 Sep: small 74M, base 194M, multi 287M; `gliner2-large-v1` is the older generation).
+ Zero-shot reference ladder (inference only, no training, descriptive context) - gliner2.5-small-v1, gliner2.5-base-v1, gliner2.5-multi-v1, gliner2-large-v1. The ladder mixes encoder, language coverage and model generation, so it does not isolate a size effect; it is not described as one. Same scorer, thresholds chosen on dev by the frozen rule, measured latency and peak memory for the Pareto figure. Protocol frozen before any test result is inspected. Coverage: 4 x (6 + 1) = 28 model-dataset pairs, 56 dev/test evaluations, not counted as training runs. The student stays multi whatever the ladder shows; the paper reports the gap.
+ Second language (secondary, exploratory) - MASSIVE slot JSON in fr-FR. MASSIVE locales are parallel localizations of the same SLURP utterances with shared ids and the same partitions (release 1.1 leaves en-US and fr-FR unchanged; https://github.com/alexa/massive). Slot values may be translated, replaced or unchanged. Arms: min vs random x {ground truth, Gemma} x N 400 x 3 seeds = 12 runs.
  + Quality - the MASSIVE authors report translation and localization noise; substring validity does not prove correct slot types, boundaries or completeness. A small French slot-quality check is reported as a diagnostic; the pool is never filtered on it.
  + Leakage controls - partitions kept as released for every locale; French selection uses French text and French student confidence only; caches namespaced by locale; English pool labels never guide French acquisition; English test labels and results never guide French choices.
  + Frozen French policy - the en-US tuning recipe (and its paid ground-truth sample) is reused unchanged and declared; the French evaluation threshold is chosen on the fr-FR dev set by the same frozen rule; early stopping on fr-FR dev; the Gemma prompt is the frozen en-US prompt with the French text, no French prompt tuning.
  + Statistics - French is excluded from both four-test Holm families, from primary pooling and from default selection. Reported as paired interaction estimates with seed spread, "does the pattern hold in a second language", not a conclusive replication or a multilingual benchmark.
  + The section 11 teacher ladder stays English-only.
+ Tuning - the section 12 Optuna searches run for gliner2.5-multi-v1; LoRA targets use the module names of the mDeBERTa encoder (checked at O5).
+ Compute - the timing gate measures multi-based training throughput and peak memory directly (1.48x parameters does not imply 1.48x runtime, most extra parameters are the embedding table). Section 6 scope covers 339 runs, 80 trials, French teacher labelling (about 11.5k pool utterances) and the reference ladder.
+ Cut order additions - the French block and the reference ladder on test sets are cut first, before item (1) of section 5; the ladder's dev evaluations stay if the Pareto figure is kept.

Run count: 306 + 15 + 6 + 12 = 339.

## 14. Data sources, decided (25 Sep 12:33, from docs/research/2026-09-25-1230-dataset-sources.md)

+ CleanCoNLL - built with the official script at `flairNLP/CleanCoNLL` `bedc569f`, which downloads CoNLL-03 from the `data.deepai.org` mirror itself. We release only sentence ids and offsets, never Reuters text. Abhishek may switch to the NIST route before release. 13,957 / 3,233 / 3,427 sentences.
+ BC5CDR - `tner/bc5cdr` `f68cdc7d` sentence split kept for comparability (known defect: some sentences split at decimal points; documented). Document ids (PMIDs) mapped from the original `CDR_Data.zip`; 6 truncated rows take the PMID of their neighbours. 5,228 / 5,330 / 5,865 sentences, 500 documents per split.
+ MIT Movie - Galileo fix `bf6c430a`, 9,775 train / 2,443 test, no dev (seeded 10% of train, as in section 2). The repo's thesis `data/mit-movie/` is the original MIT labels and its `dev.json` is an exact copy of `test.json`; any comparison with thesis numbers says so.
+ CrossRE - `mainlp/CrossRE` `a58885fa`. 1,317 of 5,265 sentences have no relation (so relation-free sentences are in the pool, as section 2 requires). The news domain is CoNLL-03 text and overlaps CleanCoNLL; stated.
+ Hallmarks - `bigbio/hallmarks_of_cancer` parquet `b78c5a2c`; 11 rows with an empty label list are treated as "none" (empty set).
+ MASSIVE 1.1 - official tarball (sha256 in the sources note). en-US and fr-FR ids, partitions and intents identical; 433 fr-FR ids have different slot types from en-US (localization).
+ Licence gates unchanged: CleanCoNLL (no licence file) and Galileo MIT Movie ("unknown") stay behind the release gate.

## 15. CrossRE with given entities, and teacher prompt v2 (Codex round 25, 25 Sep 13:34)

Pilot (kept as a pilot result): end-to-end CrossRE triples scored 0.07 micro-F1 for Gemma 4 12B (200 dev) and 0.05 for GLiNER2.5-multi zero-shot (32 dev). Too low for an active-learning comparison.

+ Task - adapted CrossRE protocol: relation prediction conditional on supplied entity mentions. Both teacher and student get the sentence, the mention offsets, ids and entity types. Candidates = all ordered pairs of distinct mentions, including pairs with no relation (the official baseline keeps only relation-bearing pairs and the first label; we do not copy that, because it reveals relation existence). Multi-label per pair; "none" = empty set; `related-to` is a positive class.
+ Student - one gliner2 `ClassificationSchema().multi("relations", labels)` decision per ordered pair, scored with `Classifier.batch_score`; head and tail are marked inside the full sentence. Training: one classification example per pair, `true_label=[]` for negatives. Budgets still count sentences ({100, 200}); a selected sentence brings all its pairs.
+ Confidence - c(x) = min over pairs and labels of |p - 0.5|, including negative pairs. Sentences with fewer than two mentions: c = 0.5, kept in the pool, ties by seed. A boundary-distance heuristic, not calibrated correctness; pair counts are reported because the minimum favours sentences with more pairs.
+ Scoring - micro-F1 over exact (head mention id, tail mention id, relation type), true negatives excluded; macro-F1 secondary (CrossRE reports macro). Text-order convention for non-directional relations as in CrossRE.
+ Supplied mentions are task input, not hidden relation labels; they are gold annotation assistance, disclosed with provenance. Candidates are never chosen from gold relations.
+ Joint NER-relation block (section 3, 30 runs) dropped: giving entities defeats it. Run count 339 - 30 = 309, plus 80 tuning trials and 56 reference evaluations.
+ Teacher prompt v2 - adds one short definition per label from `configs/label_definitions.yaml` (sources: CoNLL-2003 annotation categories and CleanCoNLL; BioCreative V CDR guidelines; CrossRE guidelines; Baker et al. 2016; MASSIVE/SLURP schema). MIT Movie definitions are proposed glosses, not official guideline text; the paper says so. CrossRE also gets its shared rules (explicit textual evidence, several labels allowed, `related-to` exclusive, no pronoun substitution or redundant chains, `named + origin` for portrayal, `part-of + role` for team or band membership).
+ Prompt gate (results in docs/research/2026-09-25-1403-teacher-prompt-gate.md: v2 for every dataset except Hallmarks) - per dataset, v1 vs v2 on the same Gemma dev ids (first 200 dev), keep v2 only if dev micro-F1 is higher (ties keep v1). CrossRE compares v1 and v2 inside the new pair-conditioned task. fr-FR inherits the en-US MASSIVE choice. The chosen prompt is applied unchanged to every teacher. Definitions and input protocol are part of the prompt hash.

## 16. Small teacher: Gemma 4 E4B replaces Qwen 3.8 2B (Codex round 26, 25 Sep 14:29)

+ Finding - no official Qwen 3.8 2B exists (Hub check 25 Sep: official Qwen 3.8 = 27B, Flash-Next 180B, 2.4T-A95B; "Qwen3.8-2B" repos are community distills). Section 11's small teacher and its Q4 pair rested on a model that does not exist.
+ Replacement - Gemma 4 E4B, official `google/gemma-4-E4B-it-qat-q4_0-gguf`, local llama.cpp, same frozen prompts. Reported as 4.5B effective parameters, about 8B including embeddings; the Pareto bubble uses 8B with the counting convention stated, and omitted modality encoders are disclosed. Latency and memory are measured, not derived.
+ Q4 pair - Gemma 4 E4B vs Gemma 4 12B, both local. The main 12B artifact (UD-Q4_K_XL GGUF of the QAT weights) is kept in both ladders A and B, so B reuses the existing Gemma cells. Quantization scheme (Q4_0 vs UD-Q4_K_XL) and architecture (E4B uses per-layer embeddings) differ: stated as unmeasured confounds. Q4 stays a descriptive checkpoint comparison. An official 12B Q4_0 run is not added (optional, predeclared, separate cache, only if added before ladder results are inspected).
+ Ladder B - the "Qwen 2B" cells become Gemma 4 E4B cells (random, N 400, 3 seeds, CleanCoNLL and Hallmarks = 6 runs). Block B stays 15 runs; total stays 309 (+80 tuning trials, +56 reference evaluations). The section 5 NER Qwen 27B block is unchanged.
+ Section 15 protocol holds in ladder A (supplied CrossRE mentions, all ordered pairs, frozen dev-selected prompts). The teacher ladder stays English-only (section 13). The replacement is frozen before any ladder test result is inspected.
+ API teachers (Qwen 3.8 27B, gpt-oss-120b on Cerebras) stay blocked until the API key is available and the hosted model ids and precision are verified in the account; hosted precision is never inferred from Hub weights. No second Qwen pair is added.

## 17. Round 27 to 30 amendments (25 Sep 22:43)

+ Training targets (round 27) - NER and slot training marks only the supplied span offsets, never every copy of the same surface text. Before the fix, 0.13% to 0.68% of pool spans also matched a non-gold occurrence. All tuning ran again under the fix; old searches are archived in `runs/tuning_archive/`.
+ Evaluation threshold (round 27) - after training, the threshold comes from dev over {0.3, 0.4, 0.5, 0.6, 0.7} (ties: closest to 0.5, then lower). The test sweep over {0.1, 0.3, 0.5, 0.7, 0.9} is descriptive only. The acquisition threshold stays 0.5.
+ CrossRE student input (round 27) - the markers carry the supplied mention types, `[H:type] ... [/H]` and `[T:type] ... [/T]`, at training and inference.
+ Process isolation (rounds 28 and 29) - every tuning trial and every matrix run starts in its own process, so no GPU memory carries over between them.
+ CrossRE negative sampling (round 30) - amends the section 15 training rule.
  + Evidence: with every ordered pair as a training example (76,450 pool pairs, 11.6% positive), five tuning trials reached dev micro-F1 0.0 to 0.035. The student predicted no relation for any pair.
  + Rule: training keeps every positive pair and `floor(r x positives)` no-relation pairs, sampled uniformly with the data-order seed. The ratio r is a CrossRE search dimension over {0.5, 1, 2, 4}, chosen on dev and frozen in the recipe for every arm.
  + The rule acts after selection and after label assignment, so teacher arms apply it to teacher labels. It never filters the pool (research rule 4).
  + Dev and test evaluation stay over all ordered pairs.
  + Each run writes its trained pair counts to `training_pairs.json`.
+ CrossRE relation head (round 32 to 34, 26 Sep) - the paper's CrossRE runs use `relation_head: native`: the checkpoint's pretrained relation scorer, applied to the given mention offsets, every ordered pair, all 17 types, BCE, LoRA on encoder + relation_scorer.
  + Why fixed by design: acquisition uses the zero-shot student's confidence. The `marker` head starts from a random layer, so its zero-shot confidence is noise and min selection would equal random selection. The native scorer is pretrained: on 200 random pool sentences its zero-shot median probability is 0.137 on true relation labels against 0.063 on others, zero-shot micro-F1 0.108, and 177 distinct confidences in 200 sentences.
  + Memory test (16 pool sentences, all negatives, 800 steps): native F1 0.906, marker 0.779, classifier 0.603.
  + A mixed-head search (archived, `runs/tuning_archive/crossre-mixed-heads/`) pruned 14 of 20 trials at step 100, including every late marker trial at dev F1 0.0. Its pick (marker, dev 0.371) is not used.
  + The native-only CrossRE search uses a MedianPruner warmup of 300 steps. The other three searches used no warmup; their recipes stay frozen.
  + `marker` and `classifier` stay in the code for the tool and for ablations.
+ Dataset pins (26 Sep) - every run and tuning search records the sha256 of every raw file the loader reads (`data.dataset_pin`). No dataset had a `source.json`, so earlier tuning pins lack dataset revisions; their split hashes are recorded.
+ Teacher span placement (26 Sep 07:10) - amends research rule 8 ("exact substring").
  + Bug: the parser placed a teacher mention at every place its text occurs, including places inside other words. On MIT Movie the rating "r" became a span inside "are", "rated" and "directors". Such spans cannot be student targets, so the training check stopped every Gemma run whose selection held one (runs failed after about 9 s). The Gemma min runs passed only because min selects mostly empty sentences.
  + Rule: a teacher mention matches only where it starts and ends on a student word edge (`gliner2` `WhitespaceTokenSplitter`). A text found only inside words is dropped and logged as `not_on_word_edges`, a teacher validation error.
  + Effect on the cached pool labels (re-parsed from the raw answers, no new teacher calls): MIT Movie 23,573 to 22,277 spans, teacher span F1 .742 to .776; CleanCoNLL 28,934 to 28,788 (.786 to .789); BC5CDR 19,557 to 19,516 (.793 to .796); MASSIVE en-US and fr-FR change by under .001.
  + Every finished run became stale through the code fingerprint and runs again. Ground-truth runs do not use the parser, and exact numerics make their reruns a reproduction check.
+ Strict exact numerics (26 Sep 08:03) - completes the exact-numerics rule.
  + Evidence: two same-seed runs of `min-N400-seed1-nopredlast` (ground truth) had equal training loss up to step 678, then drifted; in-training dev F1 differed by up to .011 and test micro-F1 by .00007. The only warning was "Memory Efficient attention defaults to a non-deterministic algorithm" (backward of `scaled_dot_product_attention` in the gliner2 heads). `warn_only=True` let it through.
  + Rule: exact numerics now calls `torch.use_deterministic_algorithms(True, warn_only=False)` with cuDNN deterministic, no benchmark, no TF32, highest fp32 matmul precision (`train.apply_exact_numerics`). An op without a deterministic kernel stops the run.
  + CrossRE: the given-mention relation training ignored exact numerics and always used bf16 autocast. It now trains in fp32 under exact numerics, like the other tasks. The CrossRE recipe was tuned in bf16, as were the NER, classification and slot recipes before exact numerics; main runs use fp32 for all four tasks.
  + Every finished run became stale again and runs again.
+ CrossRE trains every step (26 Sep 12:20) - amends the early-stopping rule for CrossRE only.
  + Evidence (fp32, ground truth, min): dev F1 moves up and down between checks. With patience 3, N100 seed 1 stopped at step 600 (best dev .337, test .309) while its trend still rose; N200 seed 1 nearly stopped at step 500 and then reached dev .410 at step 1000. The other runs trained all 1000 steps: N100 seeds 2 and 3 test .357 and .345, N200 seed 2 test .369.
  + Rule: the CrossRE recipe sets `early_stopping_patience: 10`. Runs have 10 dev checks, so every run trains all 1000 steps and keeps the checkpoint with the best dev micro-F1. Dev only, as before. Other tasks keep patience 3.
  + The recipe hash changes, so only CrossRE runs rerun. Decided by Claude with Abhishek's go-ahead ("you are in charge").
+ Implementation note (26 Sep) - Codex CLI lost its login (`401 Unauthorized`). From round 34, Claude writes the code and reviews it, with tests, until Codex is back.

## Changelog

- 2026-09-26 12:20 CEST - Section 17: CrossRE trains every step (patience 10 in its recipe), best dev checkpoint kept; CrossRE runs rerun.

- 2026-09-26 08:03 CEST - Section 17: strict exact numerics (warn_only off) and fp32 CrossRE training under exact numerics; all finished runs rerun.

- 2026-09-26 07:10 CEST - Section 17: teacher mentions match only on student word edges (parser bug fix); effect on cached labels; all finished runs rerun.

- 2026-09-26 02:37 CEST - Section 17: CrossRE relation head fixed to native by design, pruner warmup for the CrossRE search, dataset pins from raw-file hashes, Claude writes code while Codex is logged out.

- 2026-09-25 22:43 CEST - Section 17: amendments from Codex rounds 27 to 30 (exact training targets, dev threshold rule, typed CrossRE markers, process isolation, CrossRE negative sampling).

- 2026-09-25 14:29 CEST - Section 16: Gemma 4 E4B replaces the non-existent Qwen 3.8 2B; Q4 pair E4B vs 12B; counts unchanged (Codex round 26).

- 2026-09-25 13:34 CEST - Section 15: CrossRE with given entities (pair-conditioned multi-label), joint block dropped, 309 runs; teacher prompt v2 with label definitions and dev gate (Codex round 25).
- 2026-09-25 12:11 CEST - Section 13: student gliner2.5-multi-v1 (Abhishek), zero-shot reference ladder, MASSIVE fr-FR (12 runs); 339 runs. Codex round 20 changes applied: ladder qualified, French quality, leakage, frozen policy, statistics scope, compute scope, cut order, inference counts.
- 2026-09-25 11:46 CEST - Codex round 18: tuning dataset and ground-truth cost, calibration populations, budget scope.

- 2026-09-25 11:42 CEST - Section 12 from the full thesis read (Codex round 17): dev-only tuning, calibration bins, thresholds, mixing replication (6 runs); 327 runs.

- 2026-09-25 11:39 CEST - Codex round 16: pilot wording softened and sourced; Qwen 2B/27B deployment pair accepted as descriptive with the confound stated. SIGN OFF after these edits pending the full thesis read.

- 2026-09-25 11:38 CEST - Thesis pilot finding stated in section 1; teacher ladder set to newest models (Gemma 4 31B dropped; Qwen 3.8 27B via Cerebras); block B 15 runs; 321 runs.

- 2026-09-25 11:32 CEST - Section 11: teacher ladder (Q2-Q4) and teacher-to-student block (Q5), per Codex round 14; 327 runs.

- 2026-09-25 11:31 CEST - Codex round 13 (guideline additions): SIGN OFF with wording now in section 10.

- 2026-09-25 11:13 CEST - Draft after Abhishek chose the full merge.
- 2026-09-25 11:26 CEST - Round-11 edits: dev holdouts, PMID provenance, CrossRE multi-label pairs, qualified claims, licence gates, Qwen wording.
- 2026-09-25 11:23 CEST - Dataset swap for quality and size: CleanCoNLL, BC5CDR, MIT Movie fixed, CrossRE (also joint), Hallmarks of Cancer, MASSIVE kept.
- 2026-09-25 11:18 CEST - Round-9 edits: empty-slot confidence formula, joint block cut as a whole.
- 2026-09-25 11:17 CEST - Round-8 edits: framing, MASSIVE replaces E2E, scorer rules, joint controls, 5 seeds on primary cells, 306 runs, 3-4 week timeline.
