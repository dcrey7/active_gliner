---
title: GLiNER core papers
date: 2026-09-25 23:25 CEST
author: Claude (research agent)
type: wiki
status: draft
---

# GLiNER core papers

This page covers the original GLiNER line of papers and its NER follow-ups.
It does not cover GLiNER2, relation extraction or classification papers. Other pages cover those.

Each section gives the citation, the problem, the architecture, the training recipe, the evaluation, the results, the limits, and what the paper means for our study.
Every number comes from the source. The section, table or figure is in brackets.
"Not reported" means the source does not say.

Our study, in one line: we fine-tune a GLiNER2 student (`fastino/gliner2.5-multi-v1`) with LoRA on small labelled sets for NER, classification, slots and relations, in an active learning plus LLM annotation study.

## Contents

1. GLiNER (Zaratiana et al., NAACL 2024)
2. GLiNER multi-task (Stepanov and Shtopko, 2024)
3. GLiNER-BioMed (Yazdani et al., 2025)
4. GLiNER bi-encoder, "The Million-Label NER" (Stepanov et al., 2026)
5. OpenBioNER (Cocchieri et al., Findings of NAACL 2025)
6. Otter, "What Matters When Building Universal Multilingual NER Models?" (Golde et al., 2026)
7. GliLem (Dorkin and Sirts, 2025)
8. GLiNER-BioMed for BioASQ 2025 task 6 (Mehta, 2025)
9. The urchade/GLiNER repository: README, configs and fine-tuning notebook
10. Papers found but not covered here
11. Comparison table
12. Key training lessons

---

## 1. GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer

### Citation

+ Authors - Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois (FI Group, LIPN CNRS)
+ Year and venue - 2024, NAACL 2024 (Volume 1: Long Papers), pages 5364 to 5376 (from the repo README BibTeX)
+ arXiv - 2311.08526 (v1, 14 Nov 2023)
+ URL - https://arxiv.org/abs/2311.08526 and https://aclanthology.org/2024.naacl-long.300

### Problem

Classic NER models only find a fixed set of entity types.
LLMs can find any type from an instruction, but they are large, slow and costly (Section 1).
Generative models also decode entities one token at a time, so they cannot predict many types in parallel.
GLiNER asks: can a small bidirectional encoder find any entity type, in parallel, with zero-shot skill?

### Architecture (Section 2.1, Figure 2)

+ Input - one sequence: `[ENT] type1 [ENT] type2 ... [SEP] text`. The `[ENT]` and `[SEP]` tokens start random.
+ Encoder - one bidirectional LM (DeBERTa-v3 in the main runs). Labels and text attend to each other.
+ Word vectors - the first subword of each word.
+ Label vectors - the `[ENT]` outputs, passed through a two-layer feedforward network (FFN).
+ Span vectors - `S_ij = FFN(h_i concat h_j)`, from the start and end word vectors (Equation 1).
+ Max span width - K = 12 words, to keep cost linear in text length.
+ Score - `phi(i,j,t) = sigmoid(S_ij . q_t)`, a dot product plus sigmoid (Equation 2).
+ Loss - binary cross-entropy over all span and type pairs (Equation 3, Section 2.2). A pair is positive when the span has that type in the data. All other pairs are negative.
+ Decoding - greedy span selection over spans with score above 0.5. Flat NER keeps the best non-overlapping spans. Nested NER also allows fully nested spans (Section 2.3).

### Training data and recipe (Sections 3.1 and 3.2)

+ Data - Pile-NER from UniversalNER: 44,889 passages, about 240k entity spans, about 13k distinct types. ChatGPT labelled the passages with no fixed type list.
+ Backbone - DeBERTa-v3. Sizes: GLiNER-S 50M, GLiNER-M 90M, GLiNER-L 0.3B (Table 1).
+ Non-pretrained layers - width 768, dropout 0.4.
+ Optimizer - AdamW.
+ Learning rate - 1e-5 for the encoder, 5e-5 for the new layers (FFN, span layer).
+ Steps - at most 30k.
+ Warmup and scheduler - 10% warmup, then cosine decay.
+ Batch size - not reported.
+ Negative sampling - negative types are random types from other examples in the same batch.
+ Regularisation - shuffle the entity order, and randomly drop entity types (from GoLLIE).
+ Max types per sentence - 25.
+ Cost - GLiNER-L trains in 5 hours on one A100.
+ Supervised runs (Section 4.3) - same setup, up to 10,000 samples per dataset from 20 NER datasets.

### Evaluation

+ OOD benchmark - 7 datasets from CrossNER and MIT (Movie, Restaurant, AI, Literature, Music, Politics, Science) (Table 1).
+ 20 NER datasets - biomedical, news, tweets and more (Table 2).
+ Multilingual - MultiCoNER, 11 languages (Table 3).
+ Metric - entity-level F1 with exact match (Section 3.4).

### Headline results

+ OOD zero-shot average F1 (Table 1) - GLiNER-L 60.9, GoLLIE-7B 58.0, UniNER-13B 55.6, GLiNER-M 55.4, GLiNER-S 52.7, ChatGPT 47.5.
+ 20 datasets zero-shot average F1 (Table 2) - GLiNER-L 47.8, UniNER-7B 45.7, ChatGPT 36.5. GLiNER-L is best on 13 of 20 datasets.
+ Weak spot - tweets. TweetNER7 41.4 and Broad Tweeter 61.2, below UniNER (Table 2).
+ Multilingual (Table 3) - GLiNER-Multi (mDeBERTa-v3-base) averages 32.9 F1, ChatGPT 29.9, GLiNER-En 23.6. GLiNER-En gets 0.89 F1 on Bengali.
+ Supervised in-domain (Table 4) - GLiNER-L with Pile-NER pretraining 82.9, without 82.1, InstructUIE 81.2, UniNER-7B 84.8.
+ Backbones (Section 5.1, Figure 4) - DeBERTa-v3 is best. ELECTRA and ALBERT follow. BERT and RoBERTa are lower. XLNet reaches at most 3 F1 on OOD.
+ Pretraining and data size (Section 5.2, Figure 5) - Pile-NER pretraining always helps. The gain is 5.6 F1 at 100 samples per dataset and shrinks as data grows.
+ Negative type ratio (Table 5) - 0% gives P 49.3, R 58.1, F1 53.3. 50% gives P 62.3, R 59.7, F1 60.9. 75% gives P 61.1, R 56.5, F1 58.6.
+ Entity type dropping (Section 5.3, Figure 6) - over 1.4 F1 average gain out of domain.

### Limitations

+ Weaker on informal and noisy text such as tweets (Section 4.1).
+ Weak on non-Latin scripts with the English backbone (Section 4.2).
+ Behind UniNER-7B by about 3 points in supervised in-domain training (Section 4.3).
+ Max span width 12 words.
+ The uni-encoder input grows with the number of labels (later papers stress this, see Sections 3 and 4 below).

### What it means for us

+ The split learning rate (encoder low, new layers higher) is the base recipe of the whole family. Our LoRA runs should keep a separate, higher rate for the heads.
+ Negative types matter. Train with about half negative types. Too few negatives cost precision. Too many cost recall (Table 5).
+ Random label dropping and label shuffling help zero-shot transfer. Keep them when we fine-tune on small sets.
+ Pretraining gives the most gain when labelled data is small (Figure 5). This supports our small-budget design: the pretrained GLiNER2 student starts strong.
+ The 0.5 threshold is a paper default. We tune the threshold on dev, never on test.

---

## 2. GLiNER multi-task: Generalist Lightweight Model for Various Information Extraction Tasks

### Citation

+ Authors - Ihor Stepanov, Mykhailo Shtopko (Knowledgator Engineering)
+ Year and venue - 2024, arXiv preprint (no venue given in the paper)
+ arXiv - 2406.12925 (v2, 1 Aug 2024)
+ URL - https://arxiv.org/abs/2406.12925
+ Model - `knowledgator/gliner-multitask-large-v0.5` (Section 5)

### Problem

LLMs generalise, but they are costly and often fail to produce structured output (Abstract, Section 1).
The paper asks if one small GLiNER encoder can do NER, open NER, relation extraction, summarisation, question answering and open information extraction.
It also tests self-training for NER.

### Architecture (Section 2.1)

+ Backbone - DeBERTa-v3-large. Labels and text go through one encoder in one pass.
+ Token level, not span level - the model classifies tokens. This allows long outputs such as summaries.
+ BiLSTM - token vectors also pass through a bidirectional LSTM. The authors say it speeds up training and helps in low-data regimes.
+ Scoring - tokens and labels are projected to 2H. The model concatenates the token vector, the label vector and their element-wise product (3H), then an MLP gives three scores per token and class: start, end, inside.
+ Span score - the mean of the inside scores over the span.
+ Decoding - the same greedy decoding as the original GLiNER.
+ Tasks as labels - relation extraction uses labels like "source entity <> relation". Open IE uses the label "match". QA uses the label "answer" (Section 2.2).

### Training data and recipe (Sections 2.2 and 2.3)

+ Stage 1 data - synthetic data from English Wikipedia, labelled by Llama-3-8B for six tasks. Size: not reported in this paper.
+ Stage 2 data - a higher-quality mix: half synthetic, half curated NER data.
+ Stage 1 - 120,000 steps, batch size 8, lr 1e-5 encoder and 5e-5 other layers, weight decay 0.01 for both, Adam (PyTorch defaults), cosine annealing.
+ Loss - binary cross-entropy with weight 0.75 on positives and 0.25 on negatives.
+ Limits - 30 labels per example, max sequence length 768 words.
+ Stage 2 - 1,000 more steps, same batch size, lr 5e-6 encoder and 7e-6 other layers, linear scheduler.
+ Warmup - not reported.
+ Negative sampling - negatives come from other examples in the batch, which belong to other tasks (Section 3.4).
+ Self-training (Sections 2.3 and 2.4.5) - the model pre-annotates a dataset from the NER benchmark domain, then fine-tunes on it with the stage 2 learning rates. Label smoothing is added: `targets = targets x (1 - alpha) + 0.5 x alpha`.

### Evaluation (Section 2.4)

+ NER - zero-shot CrossNER (AI, Literature, Music, Politics, Science) and MIT Movie and Restaurant. Micro-F1 at span level, threshold 0.5.
+ QA - SQuAD 2.0, exact match and F1.
+ Summarisation - first 1k CNN/DailyMail examples, ROUGE-1, ROUGE-2, ROUGE-L, threshold 0.1.
+ Relation extraction - FewRel "val_wiki", exact match and F1.

### Headline results

+ NER average F1 (Table 2) - gliner-multitask-v0.5 0.6276, NuNER_Zero-span 0.6196, gliner-large-news-v2.1 0.5876, gliner_large-v2.1 0.5754. The multi-task model is weak on CrossNER AI (0.5105).
+ QA (Table 3) - EM 87.72, F1 91.99. UTC-DeBERTa-large-v2 has a higher F1 (92.53). Llama-3-8B-Instruct: EM 68.94, F1 80.51.
+ Summarisation (Table 4) - ROUGE-1 0.2484, best of all tested models.
+ Relation extraction (Table 5) - EM 82.5, F1 87.36. Llama-3-8B-Instruct: EM 38.28, F1 44.28.
+ Self-training (Table 6) - one round lifts the average NER F1:
  + gliner-multitask-large - 0.6276 to 0.6416 (500 steps, alpha 0.75, gamma 0, lr 5e-6 and 7e-6, label smoothing 0.2)
  + gliner_large-v2.1 - 0.5754 to 0.59237 (1000 steps, lr 5e-6 and 5e-6, label smoothing 0.01)
  + NuNER_Zero-span - 0.6196 to 0.6295 (100 steps, lr 5e-6 and 5e-6, label smoothing 0.01)
+ Self-training helps most where the model starts weak. CrossNER AI goes from 0.5105 to 0.6325. Where F1 is already above 0.6, there is no change or a small drop (Section 3.5).
+ The student beats its teacher, Llama-3-8B-Instruct, on QA and relation extraction (Tables 3 and 5, Section 4).

### Limitations

+ The size of the synthetic dataset is not reported.
+ Negatives come from other tasks in the batch, so they are easy. The authors expect better results from hard negatives (Section 3.4).
+ The paper does not report seeds, variance or significance tests.
+ The RE test is not end-to-end. The head entity is given in the prompt.

### What it means for us

+ It is the first GLiNER paper to show the "LLM teacher, small encoder student" loop. The student can beat the teacher. This is the core idea of our study.
+ The second-stage learning rates (5e-6 encoder, 5e-6 to 7e-6 heads) are a good reference for fine-tuning a pretrained checkpoint on small data.
+ Self-training gains are largest on weak domains and can hurt strong ones. Our active learning selection should measure gains per domain on dev.
+ Label smoothing and a 0.75 positive weight are cheap knobs for noisy LLM labels. We can tune them on dev.
+ Easy in-batch negatives are a known weakness. Hard negative types (similar labels) are worth testing.

---

## 3. GLiNER-BioMed: A Suite of Efficient Models for Open Biomedical Named Entity Recognition

### Citation

+ Authors - Anthony Yazdani, Ihor Stepanov, Douglas Teodoro (University of Geneva, Knowledgator)
+ Year and venue - 2025, arXiv preprint. A version appears in the journal Bioinformatics (article btag322), per the Oxford Academic listing.
+ arXiv - 2504.00676 (v2, 20 May 2025)
+ URL - https://arxiv.org/abs/2504.00676
+ Code and data - https://github.com/ds4dh/GLiNER-biomed

### Problem

Biomedical NER has special words, a very large number of entity types, and new types all the time (Section 1).
Fixed-label models do not generalise, and general GLiNER models are not tuned to biomedicine.

### Architecture (Section 3.3, Appendix E)

+ Uni-encoder - the standard GLiNER. Text and entity types go through one encoder. Backbones: DeBERTa-v3 small, base, large (Appendix E.1).
+ Bi-encoder - two encoders, one for text and one for entity types. Label vectors do not depend on the text, so the model can cache them. Span-to-type scoring is the same as in the uni-encoder (Section 3.3.2).
  + Text encoder - DeBERTa-v3 (small, base, large).
  + Label encoder - all-MiniLM-L6-v2 (small), bge-small-en-v1.5 (base), bge-base-en-v1.5 (large) (Appendix E.2).
+ Loss - not restated in the paper. It uses the GLiNER framework.

### Training data (Sections 3.1 and 3.2)

+ Pre-training corpus - PubMed abstracts, ClinicalTrials.gov, DailyMed labels, WIPO patents. After quality filters, TF-IDF deduplication (cosine above 0.9) and stratified sampling: about 115,000 passages, equal share per source.
+ Annotation by distillation - OpenBioLLM-70B labels 10,000 passages with 4-shot prompts. spaCy noun phrases are given as candidate entities. JSON output is forced with guided decoding. Then OpenBioLLM-8B is fine-tuned with LoRA on those 10,000 samples and labels the other 105,000 passages (Section 3.1.5).
+ Synthetic pre-training set - 105,000 samples, 2.3 million mentions, 640,000 unique entities (Section 3.1.6).
+ Post-training set - 19,000 instances, 337,000 mentions, 12,700 unique labels. It holds the 5,000-instance base set from GLiNER multi-task (1,878 curated examples from WNUT2017, OntoNotes5, MultiNERD, and 3,122 synthetic Wikipedia examples by Llama-3-8B), plus 14,000 new FineWeb examples labelled by Qwen2.5-72B (Section 3.2).
+ Bi-encoder models are also pre-trained on the NuNER corpus, because they have more parameters (Section 3.3.3).

### Training recipe (Appendix E.3)

+ Split - 90% train, 10% validation for both stages.
+ Pre-training - 20,000 steps, batch size 8, AdamW, weight decay 0.01, lr 1e-5 encoder and 5e-5 other parameters.
+ Post-training - 10,000 steps, batch size 4, AdamW, weight decay 0.01, lr 5e-6 encoder and 1e-5 other parameters.
+ Warmup, scheduler, negative sampling, label dropout, max span width - not reported.
+ Few-shot fine-tuning settings (Section 4.3) - not reported.

### Evaluation (Section 4)

+ Eight human-labelled biomedical datasets: TAC, CADEC, N2C2 2018, BC5CDR, BioRED, CHIA, Biomed NER, NCBI Disease. Total 10,918 passages, 85,959 mentions, 58 entity types (Section 4.1).
+ Metrics - micro F1, macro mean F1, macro median F1.
+ Significance - one-sided Wilcoxon signed-rank test on per-passage F1.

### Headline results

+ Zero-shot, large size (Table 1) - GLiNER-BioMed 59.77 micro F1, GLiNER-v2.5 53.81, GLiNER-BioMed-bi 54.90, UniNER-7B 48.55, OpenBioLLM-70B 36.30, Qwen3-32B 37.67.
+ The student beats both LLMs that labelled its data (OpenBioLLM-70B and 8B, p < 0.001) (Section 4.2).
+ Bi-encoder by size (Table 1) - better than the uni-encoder at small (56.93 vs 52.53) and base (58.31 vs 54.37). Worse at large (54.90 vs 59.77).
+ Few-shot, large models, micro F1 (Table 2):

| Shots | GLiNER-v2.5 | BioMed | BioMed-bi |
|---|---|---|---|
| 0 | 53.81 | 59.77 | 54.90 |
| 10 | 65.93 | 66.07 | 70.39 |
| 20 | 69.15 | 71.98 | 73.07 |
| 50 | 73.52 | 73.70 | 76.02 |
| Full | 84.64 | 84.95 | 84.91 |

+ With full data, all three models converge (p > 0.05) (Section 4.3).
+ Speed (Figure 2, Section 4.4) - the bi-encoder is 39% to 63% faster with dataset labels, and 92% to 568% faster with all 127 UMLS types (RTX 3090, FP32).
+ Ablation (Table 3), micro P / R / F1:
  + GLiNER-v2.5-large, general data only - 56.19 / 51.62 / 53.81
  + plus post-training - 55.56 / 54.06 / 54.80
  + random init, post-training only - 51.53 / 53.25 / 52.38
  + random init, synthetic biomedical only - 70.08 / 30.09 / 42.10
  + synthetic biomedical, then post-training (GLiNER-BioMed) - 56.67 / 63.22 / 59.77

### Limitations (Limitations section)

+ Synthetic labels may carry bias from the generating models.
+ The eight datasets do not cover all subdomains (for example veterinary medicine and dentistry).
+ The full pipeline needs large compute.
+ No detailed qualitative error analysis.

### What it means for us

+ LLM-only labels gave high precision and very low recall (70.08 P, 30.09 R). Teacher labels can miss many entities. Our teacher error analysis should report recall, not only F1.
+ Giving the teacher candidate noun phrases is one way to push recall. This is a prompt choice, not a gold hint, so it fits rule 9 of AGENTS.md.
+ A second stage on diverse, cleaner data fixed recall (30.09 to 63.22). Mixing sources matters.
+ With 10 labelled examples, micro F1 rises by about 6 to 15 points over zero-shot (Table 2). Small labelled sets move the model a lot. This matches our budget range.
+ Their post-training lr (5e-6 encoder, 1e-5 heads, batch 4) is a sane start for small-set fine-tuning.
+ The gap closes with full data. The benefit of better starting points is a small-data effect.

---

## 4. The Million-Label NER: Breaking Scale Barriers with GLiNER bi-encoder

### Citation

+ Authors - Ihor Stepanov, Mykhailo Shtopko, Dmytro Vodianytskyi, Oleksandr Lukashov (Knowledgator Engineering)
+ Year and venue - 2026, arXiv preprint (no venue given)
+ arXiv - 2602.18487 (v1, 11 Feb 2026)
+ URL - https://arxiv.org/abs/2602.18487
+ Models - https://huggingface.co/collections/knowledgator/gliner-bi-v2-68ac5880c610ed907cd68a5a

### Problem

The uni-encoder puts all labels into the same input as the text.
Cost grows as O((n + m)^2) for n text tokens and m label tokens (Section 2).
This blocks use with thousands or millions of types, for example UMLS with over 4 million concepts.

### Architecture (Sections 2.3 to 2.7)

+ Two encoders - a label encoder (a sentence transformer) and a text encoder. Label vectors are computed once and cached.
+ Optional cross-attention fusion ("CrossFuser") - both directions, after encoding, before scoring (Section 2.3.1).
+ Span model - start and end MLPs on word vectors. Span vector: `MLP_out(ReLU([h_start_i ; h_end_(i+k)]))`, k from 0 to K-1 (Equation 8).
+ Score - span vector dot `MLP_prompt(label vector)` (Equation 9).
+ Token model (variant) - start, end, inside scores per word and class, with a threshold rule to form spans (Section 2.5). An optional span loss can be added: `L = lambda_token L_token + lambda_span L_span`.
+ Loss - focal loss (Equation 13).
+ Negative sampling - keep all positives. Keep each negative with probability rho. Global, label-wise and span-wise masking exist (Equation 14).
+ Decoding - threshold, sort, greedy accept. Flat, multi-label and nested conflict rules (Section 2.7).

### Training data and recipe (Sections 2.8 and 2.9)

+ Pre-training data - 8M FineFineWeb samples (Large, Base, Small), 10M for Edge. GPT-4o labelled all texts.
+ Post-training data - 40k higher-quality samples, sequences up to 2048 tokens.
+ Text encoders - Ettin (ModernBERT) 400M, 150M, 68M, 32M. Label encoders - bge-base-en-v1.5, bge-small-en-v1.5, all-MiniLM-L12-v2, all-MiniLM-L6-v2.
+ Pre-training - one epoch, max length 1024, focal alpha 0.7, gamma 2.0, 500k steps (Large), 250k (Base, Small), 312.5k (Edge), batch 16 (Large) or 32 (others).
+ Post-training - one epoch, max length 2048, focal alpha 0.8, gamma 2.0, sum loss reduction.
+ Optimizer - AdamW, lr 1e-5 encoder, 3e-5 other parts, weight decay 0.01 for both, gradient clip 10.0, cosine annealing, warmup ratio 0.1.
+ Other settings - span mode MarkerV0, first-subtoken pooling, max 100 types per batch, max span width 12, dropout 0.35, RNN on, type shuffling on, max negative type ratio 1.0, label smoothing 0.0.
+ The label encoders are fine-tuned jointly with the text encoder.

### Evaluation (Section 3)

+ Zero-shot NER - CrossNER plus MIT (7 sets) and 19 other NER datasets. Micro-F1 at span level, threshold 0.4.
+ Speed - one H100, batch size 1, 1 to 1024 labels, 10 forward passes per setting.

### Headline results

+ CrossNER average (Table 1) - edge 54.0%, small 57.2%, base 60.3%, large 61.5%. The uni-encoder gliner_large-v2.5 gets 60.9% (Table 3).
+ 19-dataset average (Tables 2 and 3) - bi-encoder base and large 49.7%, uni-encoder v2.5 45.7% to 47.0%.
+ Speed (Table 4) - with cached labels, gliner-bi-edge drops only 5.2% from 1 to 1024 labels. gliner_small-v2.5 drops 98.7%.
+ At 1024 labels, the bi-encoder with cached labels is 130x faster than the matching uni-encoder (Abstract, Section 3.4).
+ Base is 98% of large quality at 2.6x the speed (Section 3.1).

### Limitations (Section 4.6)

+ Weak on highly contextual datasets: HarveyNER 10.6% to 15.0%, FabNER 22.4% to 24.3%.
+ Max span width 12 limits long entities.
+ Cross-attention fusion is not studied in depth.
+ Note: the text says the uni-encoder wins on CoNLL 2003 "(65.4% vs 66.5%)" (Section 3.3). Table 2 gives 66.5% for bi-large and Table 3 gives 64.2% for uni-large. The text and tables do not agree on this point.
+ No seeds or variance reported.

### What it means for us

+ Our label sets are small (a few to a few dozen types), so the bi-encoder speed gain matters less for us.
+ The recipe confirms the family defaults: split lr (1e-5 encoder, 3e-5 heads), warmup 0.1, cosine, focal loss with alpha 0.7 to 0.8 and gamma 2.
+ The best threshold differs by architecture (0.4 here, 0.5 in GLiNER). Tune the threshold on dev for each model.
+ Late fusion (bi-encoder) gave lower calibrated scores in the Otter study (Section 6). Uncertainty scores from different architectures are not directly comparable.

---

## 5. OpenBioNER: Lightweight Open-Domain Biomedical Named Entity Recognition Through Entity Type Description

### Citation

+ Authors - Alessio Cocchieri, Giacomo Frisoni, Marcos Martinez Galindo, Gianluca Moro, Giuseppe Tagliavini, Francesco Candoli (University of Bologna, IBM Research Europe)
+ Year and venue - 2025, Findings of the Association for Computational Linguistics: NAACL 2025, pages 818 to 837
+ arXiv - not found on arXiv
+ URL - https://aclanthology.org/2025.findings-naacl.47/
+ Code - disi-unibo-nlp/openbioner

This is not a GLiNER model. It is a GLiNER competitor. GLiNER-BioMed cites it (Section 2 of that paper). It compares directly with GLiNER-large-v1.

### Problem

Biomedical NER lacks labelled data and meets new entity types all the time.
Type names alone are often ambiguous, for example "aspirin" as Drug or Chemical (Section 3).
The paper asks if natural language type descriptions, not type names, improve zero-shot BioNER.

### Architecture (Section 4.1, Figure 2)

+ Cross-encoder - BioBERT reads `[CLS] text [SEP] description [SEP]`. One pass per entity type.
+ Token score - a linear layer maps each token vector to one score for that type.
+ Negative class - an extra "no entity" class (Appendix A).
+ Output - softmax over all types plus the negative class, then argmax per token. BIO prefixes are removed (Section 5.1).
+ Loss - class-weighted cross-entropy. All weights are 1 except the negative class weight, which is a hyperparameter (Section 4.2).
+ Entity masking - with probability p, the whole target entity is masked in the input, so the model must use context and description (Section 4.2).

### Training data and recipe (Sections 4.2, 5.1, 5.2, 5.4, Table 8)

+ Pre-training data - PileNER-biomed: the biomedical part of Pile-NER, selected by LLaMA-3.1-8B-Instruct topic classification. 59K instances, 193,235 entities, 3,896 types.
+ Descriptions - LLaMA-3.1-8B-Instruct writes general descriptions for pre-training, and task-specific descriptions from five train examples per type for inference (Section 5.2).
+ Progressive type exposure - each pass uses a random subset of 15 to 25 types. Tokens of other types become "O" (Section 4.2, Figure 3).
+ Pre-training - 4 epochs, batch size 8, constant lr 2e-5 (tuned), Adam, entity masking 0.3, dynamic negative weight (# entities / # non-entity words per batch), max input 300 tokens, max description 150 tokens, linear dropout 0.5, no warmup, seed 42. About 40 hours on one A100 80GB.
+ Fine-tuning - 8 epochs, best checkpoint on validation, entity masking 0.5, fixed negative weight 0.5 or 1.0, other settings unchanged.

### Evaluation (Section 5.3)

+ Datasets - AnatEM, NCBI, JNLPBA, BC2GM, BC4CHEMD, BC5CDR, plus two rare-type sets, JNLPBA-Rare and MedMentions-Rare (Table 1).
+ Metric - strict entity-level micro-F1 (seqeval).

### Headline results

+ Zero-shot average micro-F1 (Table 2) - OpenBioNER 52.9 (110M), GLiNER-large 51.9 (459M), UniNER 49.9 (7B), GPT-4o 43.3.
+ JNLPBA-Rare (Table 2) - OpenBioNER 63.9, GLiNER-large 51.9.
+ Effect of pre-training (Figure 4, Section 7.1) - with 100 samples per dataset, pre-training adds 35.6 F1 on average. The gap shrinks with more data.
+ Description source (Table 4) - task-specific LLM descriptions beat UMLS descriptions for OpenBioNER on all five MedMentions-Rare classes.
+ Speed (Table 5) - faster than GLiNER with up to 2 types plus negative. Slower with 5 types plus negative, because it runs one pass per type.

### Limitations (Limitations section)

+ Quality depends on the descriptions. There is no formula for a good description.
+ BIO-style tagging cannot find nested entities.
+ Pile-NER labels come from ChatGPT and may carry bias or errors.
+ Cost grows with the number of types.

### What it means for us

+ Label wording is a real lever. A clear label or a short description can beat a bare type name. GLiNER2 accepts label descriptions, so we can test this on dev.
+ Entity masking is a cheap regulariser against memorising names. It is useful when we fine-tune on a small, repeated set.
+ Pre-training helps most at 100 samples. This again supports starting from a strong pretrained student.

---

## 6. Otter: What Matters When Building Universal Multilingual Named Entity Recognition Models?

### Citation

+ Authors - Jonas Golde, Patrick Haller, Alan Akbik (Humboldt Universitat zu Berlin)
+ Year and venue - 2026, Findings of EMNLP 2026 (arXiv comment)
+ arXiv - 2601.06347 (v3, 30 Aug 2026)
+ URL - https://arxiv.org/abs/2601.06347
+ Code - https://github.com/whoisjones/otter

This paper studies the GLiNER design space in a controlled way. Its cross-encoder follows GLiNER. It compares against `gliner_multi-v2.1` and `knowledgator/gliner-x-base`.
I found no paper for GLiNER-X itself. This paper names its backbone (mT5) and its training data (Euro-GLiNER-x, 12 Latin-script languages).

### Problem

Universal multilingual NER models mix many design choices: architecture, backbone, loss, data, threshold.
Prior work never tested these choices one at a time (Section 1).

### Architecture (Section 2.1)

+ Cross-encoder (GLiNER style) - text and labels in one encoder, a `[LABEL]` token per label.
+ Bi-encoder (Binder style) - separate text and label encoders, label pooled to one vector.
+ Span vector - `MLP_span(start_i + end_j + width embedding)`. Score - dot product with the label vector (Equations 3 to 7).
+ Spans are over subword tokens, not words. No word segmenter is needed (Section 2.4).
+ In-batch negatives.

### Training recipe (Section 3.1)

+ 10k steps, no early stopping, batch size 12, AdamW.
+ lr 3e-5 for backbones and MLPs. mT5 uses 1e-3.
+ Max length 512 (1024 for mmBERT). Max span length 30 subwords.
+ MLP output size 384, width embedding size 128.
+ Loss - BCE by default.
+ Final Otter - 30k steps on FiNERweb, mmBERT backbone, BCE (Section 4).
+ Warmup and scheduler - not reported in the main text.

### Evaluation

+ Holdout for design choices - 8 languages from MultiNERD and PAN-X validation splits, 500 samples each. One global threshold is chosen here (Section 3.1).
+ Final benchmarks - DynamicNER, UNER, MasakhaNER 2.0, MultiNERD, MultiCoNER v1 and v2, PAN-X. 250 test splits, each capped at 1,000 examples (Section 4).
+ Metric - micro-F1 per dataset, macro average across datasets.

### Headline results

+ Architecture and backbone (Table 1) - the cross-encoder beats the bi-encoder on average (0.484 vs 0.460). mmBERT is the best backbone (0.529 cross, 0.489 bi). mT5 is the weakest.
+ Threshold (Figure 2, Section 3.1) - the bi-encoder peaks near 0.2 to 0.3, the cross-encoder near 0.4 to 0.5. Bi-encoder gold spans get a mean probability of 0.384, cross-encoder 0.573.
+ Script (Figure 3) - non-Latin scripts peak at lower thresholds. Subword fragmentation drives this (Appendix F).
+ Label count (Figure 4) - the cross-encoder collapses beyond 250 labels, when labels no longer fit in one pass.
+ Data (Table 2) - Euro-GLiNER-x lifts the cross-encoder with mmBERT from 0.529 (PileNER) to 0.612.
+ Loss (Table 3) - BCE 0.575, focal (alpha 0.75, gamma 1.0) 0.585, dice 0.577, BCE with positive weight 10 0.530, positive weight 100 0.432, contrastive 0.046.
+ Final (Table 4) - Otter cross-encoder with mmBERT 0.489, GLiNER-x-base 0.431, GLiNER-multi-v2.1 0.427, Qwen3-32B 0.503, Gemma3-27B 0.543, GPT-5 0.347. Focal loss at 30k steps ties BCE (0.489).

### Limitations (Limitations section)

+ Staged search, not full factorial. Loss tests use one architecture, backbone and dataset.
+ The dataset comparison has confounds (size, pipeline, label granularity).
+ Benchmarks differ in span-boundary conventions.
+ The threshold is language dependent.

### What it means for us

+ Threshold choice is a first-order effect. We must tune the decision threshold on dev, per task and per model.
+ Upweighting positives hurts (Table 3). Plain BCE is enough. Do not add heavy positive weights to fix class imbalance.
+ Architecture changes score calibration. Uncertainty-based selection depends on calibration, so we should check calibration of the GLiNER2 student on dev.
+ Multilingual training data beats English-only data for multilingual transfer. Our GLiNER2 multi student should see teacher labels in the target language when a task is not English.

---

## 7. GliLem: GliNER for contextualized lemmatization in Estonian

### Citation

+ Title - shortened here. The full title is on the arXiv page.

+ Authors - Aleksei Dorkin, Kairit Sirts (University of Tartu)
+ Year and venue - 2025, NoDaLiDa/Baltic-HLT 2025 (arXiv comment)
+ arXiv - 2412.20597 (v3, 11 Jan 2025)
+ URL - https://arxiv.org/abs/2412.20597

### Problem

The Estonian rule-based analyser Vabamorf gives many lemma candidates. Its HMM picks the right one only part of the time.
The paper asks if GLiNER can pick the right candidate.

### Architecture (Section 3)

+ The GLiNER span model matches spans of subword tokens to "labels" that are lemma transformation rules (for example "remove three last letters", Table 1).
+ Only the candidate rules from Vabamorf are given as labels for each word.

### Training data and recipe (Section 4)

+ Data - Estonian UD EDT corpus (UD version 2.14). Token and lemma pairs become transformation-rule labels.
+ Base model - the multilingual pretrained GLiNER, to reuse its span representations.
+ Recipe - the GLiNER `train.py` script with default parameters. Exact values: not reported.

### Evaluation and results (Table 2)

+ Metric - lemmatisation accuracy with bootstrap 95% intervals.
+ Test accuracy - GliLem 0.977, token classification baseline 0.966, Vabamorf HMM 0.892, oracle Vabamorf 0.993.
+ Retrieval (Table 3) - lemmatisation beats stemming by about 10% on a translated DBpedia-Entity set. GliLem adds a small, steady recall gain.

### Limitations

+ The library could not batch examples with different label sets, so inference ran one example at a time (Section 5).
+ Better lemma accuracy does not translate into much better retrieval.

### What it means for us

+ GLiNER-style span-label matching works for non-NER tasks with small, per-example label sets. This supports our use of one student for slots and classification.
+ Starting from a pretrained NER checkpoint helped even for a new task.

---

## 8. Enhancing Biomedical NER using GLiNER-BioMed with Targeted Dictionary-Based Post-processing for BioASQ 2025 task 6

### Citation

+ Author - Ritesh Mehta (Georgia Tech)
+ Year and venue - 2025, CLEF 2025 Working Notes (CEUR-WS), BioASQ task 6 (GutBrainIE)
+ arXiv - 2510.08588 (v1, 3 Oct 2025)
+ URL - https://arxiv.org/abs/2510.08588

### Problem

Fine-tune GLiNER-BioMed for 13 gut-brain entity types. Then fix frequent errors, such as gene versus chemical, with dictionary rules.

### Training data and recipe (Section 3)

+ Data - GutBrainIE Platinum (111 docs), Gold (208 docs) and Silver (499 docs). Bronze (749 docs, distant supervision) is excluded.
+ Recipe - 10 epochs, evaluation each epoch, batch size 4, max length 384 tokens, lr 1e-5 encoder and 5e-5 other layers, warmup ratio 0.1, token representations not frozen, random label dropping on.
+ Which GLiNER-BioMed checkpoint - not reported.
+ Hardware - NVIDIA Quadro M4000, 8 GB.

### Results (Table 1)

+ Dev, fine-tuned baseline - micro P 0.7390, R 0.8389, F1 0.7857.
+ Dev, with post-processing - micro F1 0.8316.
+ Test, with post-processing - micro F1 0.7743.
+ The abstract says the baseline scored 0.79 micro F1 on test. Table 1 has no baseline test row.
+ Rules tuned on dev did not transfer to test (Section 5).

### What it means for us

+ This is a direct warning about tuning on a small dev set. Rules and thresholds that fit dev errors can hurt test. Our rule is right: dev for choices, test only for final scores.
+ The recipe (lr 1e-5 and 5e-5, batch 4, warmup 0.1, 10 epochs) is a working small-data fine-tune on 8 GB.

---

## 9. The urchade/GLiNER repository: README, configs, fine-tuning notebook

### Sources

+ README - https://github.com/urchade/GLiNER
+ Training config - `configs/config.yaml` and `configs/config_biencoder.yaml`
+ Notebook - `examples/finetune.ipynb`
+ Docs - https://urchade.github.io/GLiNER/training.html

### README facts

+ Architectures - uni-encoder (supports up to about 50 entity types), bi-encoder, RelEx, GLiNER Decoder, StreamingSpan.
+ Programmatic training example - `max_steps=10000`, batch size 8, `learning_rate=1e-5`, bf16.

### `configs/config.yaml` (span model defaults)

+ Backbone `microsoft/deberta-v3-small`, span mode `markerV0`, max width 12, hidden size 768, dropout 0.3.
+ Max types 100, max length 512, max negative type ratio 1.
+ 15,000 steps, batch size 2, warmup 0.05, cosine.
+ Loss - focal alpha 0.75, gamma 0 (so a weighted BCE), label smoothing 0, sum reduction, negatives 1.0, masking none.
+ lr 1e-5 encoder, 3e-5 others. Weight decay 0.1 encoder, 0.01 others. Gradient clip 10.0.
+ Type shuffling on, random label drop on. `freeze_components` null.
+ The bi-encoder config uses the same values with batch size 8.

### `examples/finetune.ipynb` (fine-tune a pretrained checkpoint)

+ Model `urchade/gliner_small`, data `urchade/synthetic-pii-ner-mistral-v1`, 90/10 split with seed 42.
+ 500 steps, batch size 8.
+ lr 5e-6 encoder, 1e-5 others. Weight decay 0.01 for both.
+ Linear scheduler, warmup ratio 0.1.
+ Focal loss alpha 0.75, gamma 2.
+ Inference threshold 0.5. The notebook notes that v2.1 models work better with capitalised labels.

### Docs advice (training page)

+ Fine-tuning a pretrained model is almost always better than training from scratch.
+ Freeze the encoder when you fine-tune on small datasets.
+ Define similar entity types as explicit hard negatives.
+ Use focal loss for imbalanced data.
+ `start_idx` and `end_idx` are inclusive token indices.

### What it means for us

+ The fine-tuning defaults (5e-6 encoder, 1e-5 heads, 500 steps, warmup 0.1, focal gamma 2) match the second-stage values in Sections 2 and 3. This is a consistent small-data recipe across the family.
+ "Freeze the encoder on small data" supports our LoRA choice: we train a small number of encoder parameters plus the heads.
+ The docs defaults differ from the config file (for example lr_others 5e-5 vs 3e-5, dropout 0.4 vs 0.3). We must pin the library version and log the exact values we use (AGENTS.md rule 7).

---

## 10. Papers found but not covered here

I found these on arXiv. They are out of scope for this page, or I did not read them.

+ GLiNER2 (Zaratiana et al., arXiv 2507.18546) - covered on another page.
+ GLiNER-Relex (arXiv 2605.10108) - relation extraction, covered elsewhere.
+ GLiClass (arXiv 2508.07662) - classification, covered elsewhere.
+ GLiNER Guard (Minko et al., arXiv 2605.05277) - safety and PII, built on GLiNER2 (Section 3.1 of that paper). Skimmed only.
+ GLiGuard (Zaratiana et al., arXiv 2605.07982) - listed in the GLiNER README. Not read.
+ GLiNER2-PII (Zaratiana et al., arXiv 2605.09973) - built on GLiNER2. Not read.
+ Local Obfuscation by GLiNER for PII removal (arXiv 2510.19346) - application paper. Not read.
+ NERCat (arXiv 2503.14173) - Catalan fine-tuning. Not read.
+ GLiNER-X and multilingual GLiNER - I found no standalone paper. Only model cards, and the Otter comparison (Section 6).

---

## 11. Comparison table

| Paper | Architecture | Training data size | lr (encoder / other) | Steps or epochs | Main result |
|---|---|---|---|---|---|
| GLiNER (2023, NAACL 2024) | Uni-encoder, span model, DeBERTa-v3, BCE | Pile-NER: 44,889 passages, 240k spans, 13k types | 1e-5 / 5e-5 | up to 30k steps, 10% warmup, cosine | OOD avg F1 60.9 (L), beats ChatGPT 47.5 and UniNER-13B 55.6 (Table 1) |
| GLiNER multi-task (2024) | Uni-encoder, token model plus BiLSTM, DeBERTa-v3-large | Llama-3-8B synthetic Wikipedia (size not reported), then a 50/50 mix | Stage 1: 1e-5 / 5e-5. Stage 2: 5e-6 / 7e-6 | 120k steps (batch 8), then 1k steps | NER avg F1 0.6276 (Table 2). Self-training up to 0.6416 (Table 6) |
| GLiNER-BioMed (2025) | Uni- and bi-encoder, DeBERTa-v3 plus MiniLM or BGE | 105k synthetic biomedical, then 19k post-training | Pre: 1e-5 / 5e-5. Post: 5e-6 / 1e-5 | 20k steps (batch 8), then 10k steps (batch 4) | Zero-shot micro F1 59.77 vs 53.81 for v2.5. Bi-encoder 70.39 at 10-shot (Tables 1, 2) |
| GLiNER bi-encoder (2026) | Bi-encoder, Ettin (ModernBERT) plus BGE or MiniLM, focal loss | 8M to 10M GPT-4o labelled texts, then 40k | 1e-5 / 3e-5 | 250k to 500k steps (1 epoch), then 1 epoch | CrossNER 61.5% (large). 130x faster at 1024 labels (Tables 1, 4) |
| OpenBioNER (NAACL Findings 2025) | BioBERT cross-encoder with type descriptions, token model | PileNER-biomed: 59K instances, 3,896 types | 2e-5 (single, constant) | 4 epochs pre-training, 8 epochs fine-tuning | Zero-shot avg 52.9 vs GLiNER-large 51.9 (Table 2) |
| Otter (EMNLP Findings 2026) | Cross- and bi-encoder over subword spans, mmBERT | FiNERweb (91 languages), size not read | 3e-5 (all) | 10k steps (design study), 30k (final) | Macro F1 0.489 vs GLiNER-x-base 0.431 (Table 4) |
| GliLem (2025) | GLiNER span model, rules as labels | Estonian UD EDT | train.py defaults (values not reported) | not reported | Lemma accuracy 0.977 vs 0.892 (Table 2) |
| BioASQ task 6 (2025) | Fine-tuned GLiNER-BioMed | GutBrainIE Platinum, Gold, Silver (818 docs) | 1e-5 / 5e-5 | 10 epochs, batch 4, warmup 0.1 | Dev micro F1 0.7857, test 0.7743 with post-processing (Table 1) |

---

## 12. Key training lessons

1. Use two learning rates. The encoder gets about 1e-5 in pre-training and 5e-6 in fine-tuning. The new layers get 3x to 5x more (Sections 1, 2, 3, 4, 9).
2. Lower the rates in the second stage. Every paper with two stages drops the encoder rate to 5e-6 (Sections 2, 3, 9).
3. Use warmup of 5% to 10% and a cosine or linear decay (Sections 1, 4, 8, 9).
4. Train with negative types. About 50% negatives gives the best F1 in GLiNER (Table 5). Hard negatives are a known gap (Section 2).
5. Shuffle and randomly drop labels during training. Dropping gave over 1.4 F1 out of domain (Section 1).
6. Do not upweight positives heavily. It hurt F1 in Otter (Table 3). Plain BCE or mild focal loss is enough.
7. Pre-training helps most with little data. The gain was 5.6 F1 (GLiNER) and 35.6 F1 (OpenBioNER) at 100 samples per dataset.
8. Ten to fifty labelled examples move the model a lot. GLiNER-BioMed-bi rose from 54.90 to 70.39 F1 with 10 examples (Table 2).
9. LLM-only labels can have high precision and low recall (70.08 P, 30.09 R in GLiNER-BioMed Table 3). Check teacher recall.
10. A small student can beat its LLM teacher (GLiNER multi-task Tables 3 and 5, GLiNER-BioMed Table 1). Otter stays within 5.4 points of its teacher Gemma3-27B (Table 4).
11. Self-training helps weak domains and can hurt strong ones (GLiNER multi-task Section 3.5).
12. The decision threshold is a first-order choice. It depends on architecture, script and task (Otter Figures 2, 3). Tune it on dev only.
13. Rules or thresholds fitted to a small dev set may not transfer to test (BioASQ Table 1).
14. Label wording and descriptions change results (OpenBioNER Table 4, GLiNER notebook note on capitalised labels).
15. Pin the library version and log every hyperparameter. Repo docs and config defaults differ (Section 9).

## Changelog
- 2026-09-25 23:25 CEST - Created from a full read of the papers.
