---
title: Relation extraction with GLiNER-style models, and CrossRE
date: 2026-09-25 23:25 CEST
author: Claude (research agent)
type: wiki
status: draft
---

# Relation extraction with GLiNER-style models, and CrossRE

This page collects what the papers say about relation extraction (RE) with small encoders.
It ends with options for our CrossRE student and what they mean for the design.

Terms used on this page:

+ Pair - an ordered pair (head mention, tail mention) in one sentence.
+ NA - "no relation". The pair has none of the relation types.
+ Given mentions - the dataset supplies the entity spans. The model only decides relations.
+ Marker - extra text or tokens put into the sentence around the head and the tail.
+ Entity start pooling - the pair vector is the encoder output at the two start markers, joined.

Sources were read in full on 2026-09-25 (arXiv HTML, ACL PDF, GitHub source).
Local code facts come from `gliner2` 2.0.0 in this worktree's `.venv` and the cached
`fastino/gliner2.5-multi-v1` model card and `config.json`.

---

## 1. Matching the Blanks (Baldini Soares et al., ACL 2019)

**Citation.** Baldini Soares, FitzGerald, Ling, Kwiatkowski. "Matching the Blanks: Distributional Similarity for Relation Learning". ACL 2019, pages 2895 to 2905. https://aclanthology.org/P19-1279.pdf (arXiv 1906.03158).

**Formulation.** A relation statement is a sentence plus two marked spans. The model maps it to one vector (Section 2).

**Architecture (Section 3.2, 3.3, Figure 3).**

+ Encoder: BERT-large.
+ Input options tested: no markers, extra segment embeddings, entity markers.
+ Entity markers: four reserved word pieces `[E1start] [E1end] [E2start] [E2end]` around the two mentions. They are new reserved tokens, not plain text.
+ Output options tested: `[CLS]`, max pool over mention tokens, entity start.
+ Entity start: the relation vector is the concatenation of the final hidden states at `[E1start]` and `[E2start]`.
+ Then one dense layer (linear or layer norm, chosen per task), then a softmax classifier.

**Loss and NA.** Standard softmax cross entropy over the relation types. Type 0 "typically denotes a lack of relation" (Section 3.1). So NA is one more class. No re-weighting is reported.

**Training recipe (Section 3.3).** Adam, lr 3e-5, batch 64, 1 to 10 epochs (supervised). The MTB pre-training uses lr 3e-5, batch 2,048, 1 million steps, 600 million statement pairs from Wikipedia, and a blank rate of 0.7 (Sections 4.2, 5).

**Results (Table 1, dev and test F1).** Entity markers with entity start output win on all four tasks.

| Input | Output | SemEval dev | KBP37 dev | TACRED dev | FewRel 5-way-1-shot acc |
|---|---|---|---|---|---|
| Standard | `[CLS]` | 71.6 | 41.3 | 23.4 | 85.2 |
| Standard | mention pool | 78.8 | 48.3 | 66.7 | 87.5 |
| Entity markers | `[CLS]` | 81.2 | 68.7 | 65.7 | 85.2 |
| Entity markers | mention pool | 80.4 | 68.2 | 69.5 | 87.6 |
| Entity markers | entity start | 82.1 | 70.0 | 70.1 | 88.9 |

Test F1 for BERT-EM: SemEval 89.2, KBP37 68.3, TACRED 70.1. With MTB pre-training: 89.5, 69.3, 71.5 (Table 4).

**Limitations.** One pair per input. The paper does not study sentences with many candidate pairs.

---

## 2. PURE: A Frustratingly Easy Approach (Zhong and Chen, NAACL 2021)

**Citation.** Zexuan Zhong, Danqi Chen. "A Frustratingly Easy Approach for Entity and Relation Extraction". arXiv 2010.12812. https://arxiv.org/abs/2010.12812

**Formulation (Section 3.1).** For every pair of spans, predict one relation type or NA (written as epsilon).

**Architecture (Section 3.2).**

+ Two separate encoders: an entity model and a relation model.
+ The relation model runs once per pair. It inserts typed markers `<S:type> ... </S:type>` and `<O:type> ... </O:type>` around subject and object.
+ Pair vector: concatenation of the output states at the two start markers `<S:type>` and `<O:type>`.
+ Classifier: a linear layer and softmax over relation types plus NA (Appendix B).
+ The paper does not say in the text whether markers are new vocabulary items. Zhou and Chen (Section 3 below) describe this technique as "new special tokens".

**Loss and NA.** Cross entropy over R plus NA, summed over all ordered pairs of gold entities (Section 3.2, "Training and inference"). No re-weighting or negative sampling is reported. Training uses gold entities only. Training on predicted entities gave no gain (Section 5.3).

**Training recipe (Appendix B).** Relation model: 10 epochs, lr 2e-5, batch 32, Adam, linear schedule, warmup 0.1. Context window W = 100 for the relation model.

**Datasets.** ACE05, ACE04, SciERC (Table 2).

**Results.**

+ Table 1 (test, relation F1 "Rel", single sentence): ACE05 BERT-base 66.7, ALBERT 69.0; SciERC SciBERT 48.2. Cross-sentence: ACE05 BERT-base 67.7, SciERC 50.1.
+ Table 4 (dev relation F1, gold entities given, the setting closest to ours):

| Input variant | ACE05 gold | ACE05 e2e | SciERC gold | SciERC e2e |
|---|---|---|---|---|
| Text (shared span states, no markers) | 67.6 | 61.6 | 61.7 | 45.3 |
| Markers (untyped) | 70.5 | 63.3 | 68.2 | 49.1 |
| Typed markers | 72.6 | 64.2 | 69.1 | 49.7 |

With gold entities, typed markers beat the no-marker "Text" variant by 5.0 and 7.4 points.

+ Table 7 (ACE05, less data): this table compares the pipeline model with a joint model, not markers with no markers. With 10% of the training data, the pipeline gets 46.9 relation F1 and the joint model gets 37.0. It does not measure the marker effect.
+ Table 3 (speed): an approximation puts all markers at the end of the sentence with tied position ids and an attention mask. It batches all pairs of one sentence. Speed-up is 11.9x on ACE05 and 8.7x on SciERC. Relation F1 drops by 1.0 and 1.2.

**Limitations.** One forward pass per pair at training time. The batched approximation is for inference only; training with it was "slightly (and consistently) worse" (footnote 9).

---

## 3. An Improved Baseline for Sentence-level RE (Zhou and Chen, AACL 2022)

**Citation.** Wenxuan Zhou, Muhao Chen. "An Improved Baseline for Sentence-level Relation Extraction". arXiv 2102.01373. https://arxiv.org/abs/2102.01373 Code: https://github.com/wzhouad/RE_improved_baseline

**Formulation (Section 2.1).** Given a sentence and one pair, predict r from R plus NA.

**Architecture (Section 2.2).** `z = ReLU(W_proj [h_subj, h_obj])`, then softmax over R plus NA.
In the released `prepro.py` and `model.py`, `h_subj` and `h_obj` are the states at the first marker token of each entity (the `@` and `#` positions).

**Marker variants (Section 2.3, Table 1, TACRED test F1).**

| Technique | Example | BERT-base | BERT-large | RoBERTa-large |
|---|---|---|---|---|
| Entity mask | `[SUBJ-PERSON] was born in [OBJ-CITY].` | 69.6 | 70.6 | 60.9 |
| Entity marker (new tokens) | `[E1] Bill [/E1] ... [E2] Seattle [/E2]` | 68.4 | 69.7 | 70.7 |
| Entity marker (punct) | `@ Bill @ ... # Seattle #` | 68.7 | 69.8 | 71.4 |
| Typed entity marker (new tokens) | `<S:PERSON> Bill </S:PERSON> ...` | 71.5 | 72.9 | 71.0 |
| Typed entity marker (punct) | `@ * person * Bill @ ... # ^ city ^ Seattle #` | 70.9 | 72.7 | 74.6 |

+ New special tokens are "randomly initialized and updated during fine-tuning".
+ On RoBERTa-large, new special tokens hurt: typed marker is 3.6 points below its punctuation form.
+ On BERT, the new-token typed marker is slightly better than the punctuation form.

**Loss and NA.** Softmax cross entropy with NA as one class (`nn.CrossEntropyLoss` in the code). No re-weighting.

**Training recipe (Section 3.1).** Adam, lr 5e-5 (BERT-base) or 3e-5 (large), warmup 10% then linear decay, batch 64, 5 epochs, median of 5 seeds.

**Results (Table 2, test F1).** RoBERTa-large with typed punct markers: TACRED 74.6, TACREV 83.2, Re-TACRED 91.1.

**Limitations.** One pair per input. Entity types come from the dataset.

---

## 4. GLiNER multi-task (Stepanov and Shtopko, 2024), RE part only

**Citation.** Ihor Stepanov, Mykhailo Shtopko. "GLiNER multi-task: Generalist Lightweight Model for Various Information Extraction Tasks". arXiv 2406.12925. https://arxiv.org/abs/2406.12925

**Formulation (Sections 2.2, 2.4.4).** RE is span extraction. The label is the head plus the relation, `"{head} <> {relation}"`. The model marks the tail span in the text. The prompt is "Identify the relation in the given text, highlighting the relevant entity: {text}".

**Architecture (Section 2.1).** GLiNER token classification on DeBERTa-v3-large with a BiLSTM. Each token gets start, inside, end scores per label. No pair representation exists.

**Loss and NA.** Binary cross entropy with weight 0.75 on positives and 0.25 on negatives (Section 2.3). Negatives for RE are "sampled from a batch with examples that belong to other tasks" (Section 3.4). No explicit NA class.

**Training recipe (Section 2.3).** Stage 1: 120,000 steps, batch 8, lr 1e-5 encoder and 5e-5 other, weight decay 0.01, cosine, at most 30 labels, max 768 words. Stage 2: 1,000 steps, lr 5e-6 and 7e-6, linear. Data: Llama-3-8B annotations of English Wikipedia.

**Results (Table 5, FewRel val_wiki).** gliner-multitask-large-v0.5 (440M): exact match 82.5, F1 87.36. Meta-Llama-3-8B-Instruct: 38.28 and 44.28.

**Limitations.** The head must be known and written into the label. The paper names better hard negatives as future work.

---

## 5. GLiREL (Boylan, Hokamp, Gholipour Ghalandari, NAACL 2025)

**Citation.** "GLiREL: Generalist Model for Zero-Shot Relation Extraction". arXiv 2501.03172. https://arxiv.org/abs/2501.03172 Code: https://github.com/jackboyla/GLiREL (licence CC BY-NC-SA 4.0).

**Formulation (Section 2, 3.1).** Entities come from an upstream component. The model classifies every ordered pair of given entities against M text labels, in one forward pass.

**Architecture (Section 3).**

+ Input: `t0 [REL] t1 [REL] ... [SEP] x0 ... xN` plus the start and end word index of each given entity. `[REL]` and `[SEP]` are new special tokens (Appendix A.2).
+ No markers in the text. Entity vector: `FFN(h_start concat h_end)` from first-subword word states (Eq. 2).
+ Pair vector: `FFN(e_u concat e_v)` for all u not equal to v (Eq. 3). Self pairs are excluded.
+ Label vector: FFN over the first subword state of each label (Eq. 1).
+ Optional refinement: cross-attention between pairs and labels, then self-attention, at most two layers (Section 3.4).
+ Score: `sigmoid(pair dot label)` (Eq. 8). Encoder: DeBERTa-v3-large, 467M parameters in total (Section 5.1).

**Loss and NA.** Binary cross entropy per (pair, label) (Section 3.5).
In the code (`glirel/modules/base.py`, `glirel/model.py`), a pair with no gold relation gets label 0. That becomes an all-zero target row, so every label is a negative for that pair.
All ordered pairs are kept. The loss weights positives by `positive_weight` (default 2.0) and negatives by 1.0. Focal loss is an option in the config.
Each instance gets up to 25 labels. Extra labels are sampled negatives. Labels are shuffled and randomly dropped (Section 4.2).

**Training data (Section 3.6).** ZeroRel: 63,493 FineWeb texts, 25,619,624 annotated relations, "the majority of which are labeled NO RELATION". Mistral-7B-Instruct-v0.3 labels every entity pair. Labels that overlap the benchmarks are removed.

**Training recipe (Appendix A.5, Table 4).** AdamW, lr 1e-5 encoder and 1e-4 other layers, warmup 10%, cosine, batch 8, 20,000 steps, hidden 768, one T4 GPU. The repo `configs/config_finetuning.yaml` uses deberta-v3-small, dropout 0.4, 25 labels, `eval_threshold` 0.1.

**Results (Table 1, macro F1, zero-shot, mean of 5 splits).**

| m unseen | Model | Wiki-ZSL F1 | FewRel F1 |
|---|---|---|---|
| 5 | GLiREL | 62.80 | 81.21 |
| 5 | GLiREL + synthetic pre-training | 83.28 | 94.20 |
| 10 | GLiREL + synthetic pre-training | 83.67 | 87.60 |
| 15 | GLiREL + synthetic pre-training | 73.91 | 84.48 |
| 15 | TMC-BERT | 73.77 | 81.00 |
| 15 | GPT-4o | 41.57 | 70.70 |

+ Speed (Table 3, m = 10, T4 GPU): GLiREL 47.60 sentences per second on Wiki-ZSL. TMC-BERT 1.41.
+ Re-DocRED test (Table 6): GLiREL with gold coreference F1 54.13. With predicted coreference 25.08. DREEAM 80.73.
+ Ablations (Section 5.2): refinement layers help on FewRel (one pair per instance) and hurt on Wiki-ZSL (many pairs).

**Limitations (Section 7).** Labels share the 512-token window. The order and number of labels change the score of one label. Pair count grows as N squared.

---

## 6. GLiNER2 and GLiNER2.5 (Zaratiana et al., 2025) and the relation head

**Citation.** "GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface". arXiv 2507.18546. https://arxiv.org/abs/2507.18546 Repo: https://github.com/fastino-ai/GLiNER2

**What the paper covers.** NER, text classification, and hierarchical structures. The paper has no relation extraction method and no RE benchmark. It names GLiREL and GLiDRE only as related work (Section 1).

**Classification head (Appendix A).** Input `[P] task ([L] l1 [L] l2 ...) [SEP] text`. The logit for label i is `MLP(h_[L]i)`. Multi-label uses a sigmoid per label.
The local code agrees: in `gliner2/models/boundary/model.py`, `self.classifier(choice_states)` reads only the label marker states and trains with binary cross entropy.
So the head never reads text token states directly. Pair information in the text reaches the head only through encoder attention.

**Training recipe (Appendix B, Table 5).** 5 epochs, AdamW, lr 1e-5 backbone and 2e-5 task layers, weight decay 0.01, gradient clip 1.0, 1,000 warmup steps. Data: 254,334 GPT-4o-annotated examples (Section 3.1).

**Repo tutorial on RE.** https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/6-relation_extraction.md

+ API: `extract_relations(text, ["works_for", ...])` returns `(head, tail)` tuples per type. Types can have descriptions and per-type thresholds.
+ The model finds the head and tail spans itself. The tutorial shows no input for given mentions.
+ Relations are directional. `JointIE` adds typed endpoints and constraints (tutorial 15).

**GLiNER2.5 relation head (`enable_relations`).** The cached `gliner2.5-multi-v1` card says the checkpoint "was trained with `enable_relations=True`". Its `config.json` sets `relation_heads_per_type` 32, `relation_tails_per_type` 32, `relation_pair_cap` 64, `relation_argument_proposal_threshold` 0.2, `relation_biaffine_content` true, `directional_relation_states` true.
From `gliner2/models/boundary/relations.py` and `model.py`:

+ Each relation type has two query tokens, a head role and a tail role. The relation state is the concatenation of the two role states (`directional_relation_states`).
+ Pair proposals come from entity candidates. For each type, the top head and top tail candidates are crossed and capped. It is not an all-pairs matrix.
+ `SparseRelationScorer` input: head start, head end, tail start, tail end states, the relation state, the order sign, and the normalised distance. An MLP gives one logit. A biaffine term over mean-pooled span content is added.
+ Loss (`_relation_loss`): binary cross entropy over "sparse, gold-inclusive proposals", averaged over valid pairs. No class weights.
+ Training numbers and RE benchmark scores for GLiNER2.5: not reported.

---

## 7. GLiDRE (Armingaud and Besançon, 2025)

**Citation.** "GLiDRE: Generalist Lightweight model for Document-level Relation Extraction". arXiv 2508.00757. https://arxiv.org/abs/2508.00757

**Formulation (Section 3.1).** Document-level RE with given entities. Multi-label classification for every entity pair.

**Architecture (Section 3.2, 3.4).**

+ Bi-encoder: DeBERTa-v3-large for the text, BGE-large-v1.5 for labels. About 800M parameters.
+ Mention vector: pool of its words. Entity vector: mean of its mentions.
+ Pair vector: `FFN(h_head concat h_tail)`. Score: `sigmoid(pair dot label)`. No markers.
+ Optional localized context pooling from ATLOP.

**Loss and NA (Section 3.3, A.3.3).** Focal loss "to address the issue of class imbalance". NA is an all-negative pair. An ATLOP-style adaptive threshold class was tried and lost 1.4 F1 (Table 7). The best global threshold "consistently converges near 0.5". Per-class thresholds tuned on dev did not help on test.

**Training recipe (Section 4.2).** Batch 16, pre-training 50,000 steps, fine-tuning 10,000 steps, lr 1e-5 encoders and 1e-4 other layers, best dev checkpoint, threshold 0.5. Pre-training data: 136,404 FineWeb documents labelled by Mistral-Small-24B, 76,497 relation types.

**Results.**

+ Table 1 (Re-DocRED, micro F1, N training documents): N = 1: 24.45 (ATLOP 4.32, DREEAM 4.27). N = 10: 41.73 (DREEAM 27.07). N = 1000: 72.09 (ATLOP 71.70).
+ Table 3 (full supervision, test F1): 77.83. DREEAM 80.20. GLiREL 54.13.
+ Table 6: without pre-training 77.15.

**Limitations.** No cross-attention between labels in the bi-encoder (Section 3.2).

---

## 8. GLiNER-Relex (2026)

**Citation.** "GLiNER-Relex: A Unified Framework for Joint Named Entity Recognition and Relation Extraction". arXiv 2605.10108. https://arxiv.org/abs/2605.10108

**Formulation.** Joint NER and RE from raw text in one pass (Section 3.1).

**Architecture (Sections 3.2 to 3.6).** Input `[ENT] e1 ... [REL] r1 ... [SEP] text`. DeBERTa-v3-large plus BiLSTM. Span vectors as in GLiNER. The released model scores all ordered pairs of predicted entities. Pair vector `MLP([s_a; s_b])`, score `pair dot relation`.

**Loss and NA (Section 3.7, 3.9).** Focal loss with alpha 0.75 and gamma 0, which is alpha-balanced binary cross entropy. "Optional negative sampling". Relation threshold 0.5.

**Training recipe (Section 3.9, Table 1).** Stage 1: about 1 million Qwen3-32B sentences plus 50,000 documents, 1 epoch, batch 8, warmup 0.05. Stage 2: about 3,000 Gemini examples, 5 epochs. The HTML shows the learning rates without their leading digits, so the values are not reported here.

**Results (Table 2, zero-shot micro F1).**

| Model | Entities | CoNLL04 | DocRED | FewRel | CrossRE |
|---|---|---|---|---|---|
| GLiREL | gold | 4.5 | 2.4 | 24.0 | 1.4 |
| GLiNER2 | predicted | 32.9 | 11.7 | 20.8 | 6.0 |
| GPT-5-mini | prompted | 42.4 | 18.6 | 15.0 | 12.4 |
| GLiNER-Relex | predicted | 40.4 | 31.3 | 12.5 | 18.1 |

The Section 4.4 text gives other numbers for GLiNER2 (for example 4.9 on CrossRE) than Table 2. The table values are used here.

**Limitations (Section 5.4).** Precision drops in entity-dense text because all-pairs enumeration creates many candidates.

---

## 9. CrossRE (Bassignana and Plank, EMNLP Findings 2022)

**Citation.** Elisa Bassignana, Barbara Plank. "CrossRE: A Cross-Domain Dataset for Relation Extraction". arXiv 2210.09345. https://aclanthology.org/2022.findings-emnlp.263 Data: https://github.com/mainlp/CrossRE

**Domains and size (Section 3.2, Table 1).** Six English domains. News is CoNLL-2003 (Reuters). The other five are Wikipedia text from CrossNER.

| Domain | Sentences train / dev / test | Relations train / dev / test |
|---|---|---|
| news | 164 / 350 / 400 | 175 / 300 / 396 |
| politics | 101 / 350 / 400 | 502 / 1,616 / 1,831 |
| natural science | 103 / 351 / 400 | 355 / 1,340 / 1,393 |
| music | 100 / 350 / 399 | 496 / 1,861 / 2,333 |
| literature | 100 / 400 / 416 | 397 / 1,539 / 1,591 |
| AI | 100 / 350 / 431 | 350 / 1,006 / 1,127 |
| total | 668 / 2,151 / 2,446 | 2,275 / 7,662 / 8,671 |

A "relation" here is a directed entity pair with at least one label.

**Relation types (17).** part-of, physical, usage, role, social, general-affiliation, compare, temporal, artifact, origin, topic, opposite, cause-effect, win-defeat, type-of, named, related-to.

+ Multi-label: 6% of relations have more than one label (Section 3.2).
+ `related-to` is exclusive. It is used only when no other label fits.
+ Direction matters. (e1, e2) and (e2, e1) are different pairs.

**Label distribution (Appendix D, Table 10, summed over the six domains, 19,761 labels).** role 21.8%, physical 14.1%, general-affiliation 12.0%, part-of 8.8%, artifact 7.1%, named 6.6%, temporal 5.3%, related-to 4.6%, win-defeat 4.3%, origin 3.8%, type-of 3.6%, usage 1.9%, opposite 1.8%, topic 1.6%, compare 1.1%, social 1.0%, cause-effect 0.6%.
Domains differ a lot. Examples: news has physical 31.19% and role 31.07%; music has general-affiliation 27.5%; politics has role 37.38% and temporal 16.41%.

**Share of no-relation pairs.** Not reported in the paper. We counted it on the official files (commit `a58885f` in `~/.cache/active_gliner/raw/crossre/`). We counted all ordered pairs of distinct mention spans per sentence. Our related-pair counts match Table 1 exactly.

| Split | Ordered pairs | Related pairs | No-relation share |
|---|---|---|---|
| train | 18,934 | 2,275 | 88.0% |
| dev | 67,382 | 7,662 | 88.6% |
| test | 79,988 | 8,671 | 89.2% |

That is about 7.3 no-relation pairs per related pair in train.

**Evaluation protocol (Sections 4.1, 4.3).**

+ Task: relation classification (RC) only. Entities are given, and only pairs "identified as being semantically connected" are classified. No-relation pairs are not in the test.
+ Metrics: micro F1, macro F1, weighted F1. Macro F1 skips classes with zero support in the test set.
+ One model per domain, trained on that domain. Five seeds.

**Baseline model (Section 4.2, Appendix F Table 8).**

+ Typed markers as text: `<E1:person> Cunningham </E1:person> ... <E2:organization> Philadelphia Eagles </E2:organization>`.
+ bert-base-cased. Pair vector: the two start-marker outputs, joined. One linear layer, softmax.
+ Single label: the 6% multi-label pairs are ignored. A multi-head model was tested, "but the per-label data is not enough".
+ Adam, lr 2e-5, batch 32, cross entropy, seeds 4012, 5096, 8878, 8857, 9908.

**Baseline results (Table 4, test, mean of 5 seeds).**

| Metric | news | politics | science | music | literature | AI | avg |
|---|---|---|---|---|---|---|---|
| Micro F1 | 46.36 | 58.26 | 40.10 | 75.96 | 67.70 | 45.40 | 55.63 |
| Macro F1 | 16.52 | 20.33 | 25.29 | 39.19 | 37.74 | 30.66 | 28.29 |
| Weighted F1 | 37.59 | 53.53 | 35.84 | 73.16 | 63.08 | 41.52 | 50.79 |

Many per-class scores are 0.0 (Table 5). Examples: cause-effect in all domains, social in four domains.

**Later results on CrossRE.**

+ Silver Syntax Pre-training (Bassignana, Ginter, Pyysalo, van der Goot, Plank; arXiv 2305.11016). Same model, but the no-relation case is included: "our score range is lower because we include the no-relation case, while they assume gold entity pairs". In-domain macro F1 of the baseline (Table 1 diagonal): news 10.98, politics 11.30, science 8.57, music 19.01, literature 17.17, AI 15.57. Syntax pre-training raises the average by 0.71. This is the published setting closest to ours.
+ Multi-CrossRE (NoDaLiDa 2023; arXiv 2305.10985). Same RC setup with XLM-R large. English macro F1 average 23.3 (Table 3).
+ How to Encode Domain Information (LREC-COLING 2024). Multi-domain training, RC setup of the original paper. Test macro F1 average: baseline 36.47, special domain marker token 38.66 (Table 2). Fine-grained entity types in markers gave 34.10 on dev, below the baseline 35.48. The paper also adds "CrossRE 2.0" news data (4,590 more sentences).
+ GLiNER-Relex (Section 8 above): zero-shot, end-to-end, CrossRE micro F1 18.1. GLiREL with gold entities: 1.4.

---

## 10. Comparison table

| Method | How the pair enters the model | Pair vector | Scorer | NA handling | Pairs per forward pass | Relation pre-training |
|---|---|---|---|---|---|---|
| MTB (2019) | reserved marker tokens | states at the 2 start markers | linear + softmax | NA is class 0 | 1 | MTB, 600M pairs |
| PURE (2021) | typed markers | states at the 2 start markers | linear + softmax | NA class, all gold-entity pairs | 1 (batched approx. at inference) | none |
| Zhou and Chen (2022) | typed markers, punctuation or new tokens | states at the 2 first marker tokens | MLP + softmax | NA class | 1 | none |
| CrossRE baseline (2022) | typed markers as text | states at the 2 start markers | linear + softmax | related pairs only (RC) | 1 | none |
| GLiNER multi-task (2024) | head written into the label text | none (tail is a span) | token BCE | sampled label negatives | all labels | Llama-3-8B synthetic |
| GLiREL (2025) | given spans, no markers | FFN(start, end) per entity, FFN(pair) | dot with label, sigmoid | all-zero row, pos weight 2.0 | all pairs x all labels | ZeroRel, 25.6M relations |
| GLiNER2 classification | whatever is in the text | label token state only | MLP on `[L]` state, sigmoid | all-zero labels | 1 text | not for RE |
| GLiNER2.5 relation head | spans the model proposes | 4 endpoint states + relation role states + order + distance | MLP + biaffine, sigmoid | BCE over gold-inclusive proposals | capped pairs x types | not reported |
| GLiDRE (2025) | given entities, no markers | FFN(head, tail) | dot with label, sigmoid | focal loss, threshold 0.5 | all pairs x all labels | 136k synthetic docs |
| GLiNER-Relex (2026) | predicted spans | MLP(head, tail) | dot with label, sigmoid | alpha-balanced BCE | all pairs x all labels | 1M synthetic sentences |

---

## 11. How to predict relations over given mentions with a small encoder

The task: for every ordered pair of given CrossRE mentions, output zero or more of 17 labels.
The options below are ranked by the evidence above.

### Option 1 (best evidence): typed markers plus entity start pooling

+ Keep one marked input per pair, as now: `[H:type] ... [/H]` and `[T:type] ... [/T]`.
+ Add a small head that reads the encoder states at the first subword of the `[H:` and `[T:` markers, joins them, and applies an MLP with 17 sigmoid outputs.
+ NA is the all-zero target, as in GLiREL and GLiDRE. The CrossRE baseline used a softmax with one label per pair instead.
+ Evidence:
  + MTB Table 1: entity start beats `[CLS]` and mention pooling on all four tasks.
  + PURE Table 4: with gold entities, typed markers beat the no-marker pair vector by 5.0 (ACE05) and 7.4 (SciERC).
  + PURE Table 7 compares pipeline with joint models (46.9 against 37.0 at 10% data). It is not evidence about markers.
  + It is the CrossRE baseline model, so our numbers stay comparable to Sections 9 and to Silver Syntax.
+ Cost: one forward pass per pair. CrossRE train has about 28 ordered pairs per sentence (18,934 over 668 sentences). PURE's batched approximation cuts inference time by 8.7x to 11.9x for about 1 F1.
+ Risk: the marker text is new to the model. Zhou and Chen show that the marker form changes F1 by up to 3.6 points on one encoder. mDeBERTa is not tested in any source.

### Option 2: the GLiNER2.5 pretrained relation scorer, fed with given spans

+ Build the pair list from the given mentions (all ordered pairs) instead of the model's own proposals. Score it with `SparseRelationScorer` and the 17 relation role queries.
+ One forward pass per sentence scores all pairs and all types.
+ Strength: the scorer weights ship in the checkpoint (`enable_relations=True`). It already has an order feature and directional role states.
+ Weakness: no source reports its accuracy. It uses shared endpoint states, which is the "Text" style that PURE Table 4 ranks below markers. It needs code that reaches into `gliner2` internals, because the public API does not take given mentions.

### Option 3: a GLiREL-style pair head

+ Labels in the prompt, given spans as indices, `FFN(start, end)` per mention, `FFN(head, tail)` per pair, dot product with each label vector.
+ One forward pass per sentence.
+ Evidence: strong zero-shot numbers after 25.6M synthetic relations (Table 1). Without that pre-training, FewRel m = 5 falls from 94.20 to 81.21 and Wiki-ZSL from 83.28 to 62.80.
+ Weakness: the pair head starts from random weights. GLiNER-Relex reports GLiREL at 1.4 micro F1 on CrossRE zero-shot.

### Option 4 (current design): markers plus the GLiNER2 classification head

+ The head is `MLP(h_[L])` over the 17 label tokens (GLiNER2 Appendix A; `model.py`).
+ No paper in this survey uses a label-token head for pair RE. Every marker paper reads the marker states themselves.
+ Our run log: 16 training sentences, 1:1 negatives, 400 steps, F1 0.55. With all negatives, F1 near 0. Design section 17 records dev micro F1 0.0 to 0.035 on all 76,450 pool pairs.

### Training settings that most sources share

+ Positive up-weighting or focal loss: GLiREL positive weight 2.0; GLiNER multi-task 0.75 against 0.25; GLiNER-Relex alpha 0.75; GLiDRE focal loss.
+ Head learning rate about 10x the encoder rate: GLiREL and GLiDRE 1e-5 and 1e-4; PURE entity model 1e-5 and 5e-4.
+ Train on gold entity pairs only (PURE Section 3.2).
+ One global threshold near 0.5 was enough for GLiDRE (Appendix A.3.3).

---

## 12. What it means for us

+ **The head is the likely problem, not the markers.** Our markers match the CrossRE baseline. Our head does not. The GLiNER2 classifier reads only the `[L]` label states, so the pair signal must travel through attention under LoRA. This is a hypothesis from the code read. We have not tested it.
+ **Recommended formulation: Option 1.** Keep the typed markers. Add entity start pooling at the `[H:` and `[T:` markers with a 17-way sigmoid MLP. Keep the dev-tuned negative ratio from design section 17. Add a positive weight or focal loss, and a higher head learning rate. Choose all of these on dev.
+ **Test Option 2 on dev as a second candidate.** It is cheap at inference and pre-trained, but no one has published its accuracy.
+ **This is a design change.** AGENTS.md says to stop and say so. The design page (`docs/research/2026-09-25-1113-full-merge-design.md`, section 15 and 17) must record it before Codex writes code.
+ **Expected score range.** The closest published setting (all pairs, no-relation included, bert-base, about 100 training sentences per domain) gives in-domain macro F1 from 8.57 to 19.01 (Silver Syntax Table 1). Scores near the RC baseline (macro 28.29, micro 55.63) are not a fair target, because RC removes the 88% no-relation pairs.
+ **Report both views.** Report micro F1 over all ordered pairs, as now. Also report macro F1 that skips zero-support classes, as CrossRE does, so readers can compare.
+ **Imbalance is large.** 88.0% of ordered train pairs have no relation, and 6 of 17 labels are under 2% of all labels. Expect 0.0 F1 on rare classes, as in CrossRE Table 5.

---

## Changelog
- 2026-09-25 23:35 CEST - Claude check: corrected the PURE Table 7 reading (pipeline against joint, not markers against no markers).
- 2026-09-25 23:25 CEST - Created from a full read of the papers.
