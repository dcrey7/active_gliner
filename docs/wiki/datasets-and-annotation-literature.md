---
title: Datasets and the LLM-annotation literature
date: 2026-09-25 23:25 CEST
author: Claude (research agent)
type: wiki
status: draft
---

# Datasets and the LLM-annotation literature

This page has two parts.

+ Part A: our NER datasets, their papers, and the scores other people report on them.
+ Part B: the papers behind our research question, what each found, and the gap our paper fills.

Our question: when does student uncertainty improve one-round LLM annotation, and when does teacher error erase that benefit?

## How I read the sources

+ I read the full text of every arXiv paper below (PDF to text, 25 Sep 2026).
+ I read the BC5CDR paper through the Europe PMC full-text XML (PMC4860626).
+ I read the MIT Movie paper as a PDF from `sls.csail.mit.edu`.
+ For "Mind Your Outliers" (arXiv 2107.02331) I read only the abstract. I say so where I cite it.
+ Venues come from the arXiv "comments" field or the paper header.
+ Numbers are copied from the named table or section. "Not reported" means the source does not give the number.
+ Scores in different papers use different label sets, splits and metrics. Do not compare them without reading the "Setting" column.

---

# Part A: our datasets

## A1. CleanCoNLL

**Citation.** Susanna Rücker and Alan Akbik. "CleanCoNLL: A Nearly Noise-Free Named Entity Recognition Dataset". EMNLP 2023. arXiv 2310.16225. https://arxiv.org/abs/2310.16225

**What it is.** A relabeled English CoNLL-03 with 4 types (PER, LOC, ORG, MISC) and an added entity-linking layer.

| Fact | Value | Source |
|---|---|---|
| Labels changed versus CoNLL-03 | 7.0% of all labels; 10.3% of sentences | Abstract, Section 3 (Table 3) |
| Labels changed in the test split | 9.4% of labels; 14.2% of sentences | Section 3 |
| Sentences (all splits) | CoNLL-03 20,744; Reiss 20,617; CleanCoNLL 20,617 | Table 2 |
| Entities (all splits) | CoNLL-03 35,089; Reiss 34,941; CleanCoNLL 35,257 | Table 2 |
| Estimated annotation errors in a 100-sentence sample | about 10% in CoNLL-03, about 7% in Reiss, about 1% in CleanCoNLL | Section 3, Figure 4 |
| Share of "model errors" that are really label errors (FLERT, 100 errors each) | 47% on CoNLL-03, 6% and 7% on the two CleanCoNLL variants | Section 4.2, Figure 5 |

Our build (see `docs/research/2026-09-25-1230-dataset-sources.md`) has 20,617 sentences and 35,257 entities. Both totals match Table 2.

**Fine-tuned scores (Table 4, test F1, mean of 3 seeds, 2 for ACE).**

| Model | CoNLL-03 | CleanCoNLL |
|---|---|---|
| Flair | 92.77 | 94.21 |
| FLERT (xlm-roberta-large) | 93.94 | 96.98 |
| Biaffine (xlm-roberta-large) | 93.84 | 97.08 |
| ACE (sentence level) | 93.56 | 95.89 |

The abstract rounds the best score to 97.1%.

**Zero-shot or LLM scores on CleanCoNLL.** The CleanCoNLL paper reports none. NoiseBench (Section B1.8) reports scores on the CleanCoNLL test labels. Other papers report LLM and GLiNER scores on the original CoNLL-03 labels. See Part C.

## A2. BC5CDR

**Citation.** Jiao Li, Yueping Sun, Robin J. Johnson, Daniela Sciaky, Chih-Hsuan Wei, Robert Leaman, Allan Peter Davis, Carolyn J. Mattingly, Thomas C. Wiegers, Zhiyong Lu. "BioCreative V CDR task corpus: a resource for chemical disease relation extraction". Database 2016, article baw068. doi:10.1093/database/baw068. PMC4860626.

**What it is.** 1,500 PubMed abstracts with chemical and disease mentions, MeSH ids, and chemical-induced-disease (CID) relations.

| Split | Articles | Disease mentions | Disease ids | Chemical mentions | Chemical ids | CID relations |
|---|---|---|---|---|---|---|
| Training | 500 | 4,182 | 1,965 | 5,203 | 1,467 | 1,038 |
| Development | 500 | 4,244 | 1,865 | 5,347 | 1,507 | 1,012 |
| Test | 500 | 4,424 | 1,988 | 5,385 | 1,435 | 1,066 |

Source: Table 2. Our TNER copy has the same mention counts per split and type.

+ Inter-annotator agreement (Jaccard, Table 3): disease 0.8749, chemical 0.9605 over all sets.
+ BioCreative V task: the best disease NER (DNER) system scored F 86.46%; the best CID system scored 57.03% (Section "BioCreative V"). DNER includes concept normalisation, so it is not plain mention F1.

**BLURB use.** Yu Gu, Robert Tinn, Hao Cheng, et al. "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing". arXiv 2007.15779 (v6, 2021). https://arxiv.org/abs/2007.15779

+ BLURB splits BC5CDR into two separate NER tasks: BC5-chem and BC5-disease (Section 2.3.1).
+ BLURB uses the Crichton et al. preprocessed version and entity-level F1 (Table 3).
+ Instance counts in Table 3 (train / dev / test): BC5-chem 5,203 / 5,347 / 5,385; BC5-disease 4,182 / 4,244 / 4,424.

| Model (BLURB, Table 6) | BC5-chem F1 | BC5-disease F1 |
|---|---|---|
| BERT uncased | 89.25 | 81.44 |
| BioBERT cased | 92.85 | 84.70 |
| PubMedBERT uncased | 93.33 | 85.62 |

**Important for us.** Our BC5CDR task has both types in one model. BLURB scores each type alone. The joint-type scores in Part C come from GLiNER, UniNER and FreeAL.

## A3. MIT Movie

**Citation.** Jingjing Liu, Panupong Pasupat, D. Scott Cyphers, James Glass. "Asgard: A portable architecture for multilingual dialogue systems". ICASSP 2013. https://sls.csail.mit.edu/publications/2013/Liu_ICASSP-2013.pdf

**What it is.** Crowd-sourced spoken-style movie queries with 12 semantic classes: Title, Viewers' rating, Year, Genre, Director, MPAA rating, Plot, Actor, Trailer, Song, Review, Character (Table 2).

+ The paper reports 12,000 English movie sentences: 6,800 frame-based and 5,200 free-style (Table 3).
+ The paper uses a random 80% / 20% train / test split (Section 4).
+ The best semi-Markov CRF scores F 88.58 on movie, against a baseline of 85.84 (Table 4).
+ The public files `engtrain.bio` and `engtest.bio` hold 9,775 and 2,443 sentences (our count). The paper does not describe these files.
+ The paper gives no dev split. The public release has no dev split.

**How the zero-shot literature uses it.** CrossNER itself does not contain MIT Movie. InstructUIE and UniNER hold out "CrossNER and MIT" as an out-of-domain (OOD) benchmark (UniNER Section 5.3). GLiNER Table 1 reuses this 7-dataset OOD benchmark: Movie, Restaurant, and the 5 CrossNER domains. GLiNER2 (Table 3) reports only the 5 CrossNER domains and no MIT Movie score.

**Important for us.** We use the Galileo fix (`rungalileo/mit_movies_fixed_connll_format`). All scores in the literature use the original MIT labels. The Galileo fix changes entity sets in 280 train and 108 test sentences (our audit). So literature scores are a guide only.

## A4. CrossRE (sizes only)

**Citation.** Elisa Bassignana and Barbara Plank. "CrossRE: A Cross-Domain Dataset for Relation Extraction". arXiv 2210.09345. https://arxiv.org/abs/2210.09345

Paper Table 1 (sentences, train / dev / test):

| Domain | Train | Dev | Test | Total |
|---|---|---|---|---|
| news | 164 | 350 | 400 | 914 |
| politics | 101 | 350 | 400 | 851 |
| science | 103 | 351 | 400 | 854 |
| music | 100 | 350 | 399 | 849 |
| literature | 100 | 400 | 416 | 916 |
| AI | 100 | 350 | 431 | 881 |
| **Total** | **668** | **2,151** | **2,446** | **5,265** |

+ Paper relation totals: 2,275 / 7,662 / 8,671 (18,608). 17 relation types.
+ Our count at commit `a58885fa` gives 2,394 / 8,199 / 9,168 relations. Sentence counts match exactly. I did not find the reason for the relation gap (unconfirmed).
+ The relations page covers scores.

## A5. Our student: GLiNER2.5 Multi

+ The model card for `fastino/gliner2.5-multi-v1` (read 25 Sep 2026) reports no benchmark scores. It lists 287M parameters and an mDeBERTa-v3-base encoder.
+ The GLiNER2 paper (arXiv 2507.18546) covers an earlier checkpoint. Its Table 3 gives CrossNER zero-shot F1 (see B1.7). It reports no score on CleanCoNLL, CoNLL-03, BC5CDR or MIT Movie.

---

# Part B: the literature on our question

## B1. LLM annotation for training small models

### B1.1 Want to reduce labeling cost? GPT-3 can help

+ **Citation.** Shuohang Wang, Yang Liu, Yichong Xu, Chenguang Zhu, Michael Zeng. Findings of EMNLP 2021. arXiv 2108.13487.
+ **Question.** Can GPT-3 labels train smaller models at lower cost than human labels?
+ **Method.** GPT-3 few-shot labels unlabeled data. RoBERTa-large (NLU) or PEGASUS (NLG) trains on the labels. They also mix GPT-3 and human labels under one budget. "Active labeling" sends the GPT-3 labels with the lowest GPT-3 confidence (first-token logit) to humans for relabeling (Section 2, Figure 2d).
+ **Datasets.** 9 tasks: SST-2, CB, TREC, AGNews, DBPedia, RTE, Gigaword, SQuAD (question generation), XSum.
+ **Results.**
  + Same downstream score for 50% to 96% less cost than human labels (abstract). SST-2 saves 96%; Gigaword saves 93.8% (Section 1).
  + Students trained on enough GPT-3 labels beat GPT-3 itself (Section 3.3.2, Figure 4).
  + GPT-3 confidence tracks GPT-3 accuracy. The top 10% most confident labels have accuracy 95%, 90%, 95% on TREC, AGNews, DBPedia; low-confidence labels are much less accurate (Section 3.3.3, Figure 5).
  + Active labeling lifts TREC accuracy from 77% to 80% at a $2.2 budget (Section 3.3.3).
  + With a large budget, full human labeling wins (Section 3.3.1).
+ **Relevance.** Teacher errors concentrate where the teacher is unsure. This is teacher-side confidence, not student uncertainty. Classification only, no NER.

### B1.2 Is GPT-3 a good data annotator?

+ **Citation.** Bosheng Ding, Chengwei Qin, Linlin Liu, Yew Ken Chia, Boyang Li, Shafiq Joty, Lidong Bing. ACL 2023. arXiv 2212.10450.
+ **Question.** Which GPT-3 annotation route is best: label real data (PGDA), generate data (PGDG), or generate with a dictionary (DADG)?
+ **Method.** GPT-3 labels or generates data; BERT-base trains on it; scores are on human test labels.
+ **Datasets.** SST-2, FewRel, CrossNER (AI domain), ASTE (laptop).
+ **Results on CrossNER AI (Table 3, F1).**

  | Approach | Samples | Cost (USD) | F1 |
  |---|---|---|---|
  | PGDA (GPT-3 labels the 100 gold train sentences, 10-shot) | 100 | 15.39 | 23.08 |
  | PGDG (zero-shot generation) | 3,000 | 13.56 | 41.35 |
  | DADG (Wikidata-assisted generation) | 3,000 | 13.61 | 47.22 |
  | Human labeled | 100 | 17 to 42.85 | 42.00 |
  | PGI (GPT-3 direct inference on test) | 431 | 63.23 | 46.65 |

  + GPT-3 finds entities but also tags entities of the wrong type and misses boundaries (Section 4.3.1).
  + Direct labeling suits small label spaces; generation suits large label spaces (Section 1).
+ **Relevance.** Direct GPT-3 labeling for NER was weak in 2022-2023. Type errors and boundary errors are the main NER failure modes.

### B1.3 ChatGPT outperforms crowd-workers for text-annotation tasks

+ **Citation.** Fabrizio Gilardi, Meysam Alizadeh, Maël Kubli. PNAS 2023. arXiv 2303.15056. https://www.pnas.org/doi/10.1073/pnas.2305016120
+ **Question.** Is zero-shot ChatGPT better than MTurk crowd workers at annotation?
+ **Method.** Same codebooks for ChatGPT (temperature 1 and 0.2, two runs each), MTurk, and trained research assistants.
+ **Datasets.** 6,183 tweets and news articles; tasks: relevance, stance, topics, two frame sets.
+ **Results.** ChatGPT accuracy beats MTurk by about 25 points on average. ChatGPT intercoder agreement beats MTurk and trained annotators on all tasks. Cost is under $0.003 per annotation, about 30 times cheaper than MTurk (abstract, Figure 1).
+ **Relevance.** Motivates LLM labels. Classification only. Accuracy is measured against trained annotators, not per difficulty.

### B1.4 UniversalNER (UniNER)

+ **Citation.** Wenxuan Zhou, Sheng Zhang, Yu Gu, Muhao Chen, Hoifung Poon. ICLR 2024. arXiv 2308.03279.
+ **Question.** Can targeted distillation from ChatGPT build a small open NER model?
+ **Method.** ChatGPT (`gpt-3.5-turbo-0301`, temperature 0) labels 50K Pile passages with open entity types. After filtering: 45,889 passages, 240,725 entities, 13,020 types (Section 2). LLaMA 7B and 13B are instruction-tuned on this data.
+ **Datasets.** A 43-dataset benchmark across 9 domains.
+ **Results.**
  + Average zero-shot F1 over 43 datasets: UniNER-7B 41.7, UniNER-13B 43.4, ChatGPT 34.9 (Section 5.2).
  + OOD benchmark (Table 3): see Part C for Movie.
  + Frequency-based negative sampling adds 21.9 points over no negatives (Table 4).
  + Supervised on 20 datasets, UniNER-7B averages 84.78 F1 (Table 2).
+ **Relevance.** The student beats its LLM teacher on average. Labels are generated once over a random pool, with no selection.

### B1.5 GLiNER

+ **Citation.** Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois. arXiv 2311.08526 (v1 comment: "Work in progress").
+ **Question.** Can a small bidirectional encoder do open-type NER as well as LLMs?
+ **Method.** DeBERTa-v3 span model trained on Pile-NER (the UniNER ChatGPT labels). Negative types are sampled from the batch (Section 3.2).
+ **Results.** Zero-shot OOD average: GLiNER-L 60.9 versus ChatGPT 47.5 and UniNER-13B 55.6 (Table 1). 20-dataset zero-shot average: GLiNER-L 47.8, UniNER-7B 45.7, ChatGPT 36.5 (Table 2).
+ **Relevance.** Our student family learns from LLM labels. GLiNER is a distilled model, but the paper does no data selection.

### B1.6 NuNER

+ **Citation.** Sergei Bogdanov, Alexandre Constantin, Timothée Bernard, Benoit Crabbé, Etienne Bernard. arXiv 2402.15343.
+ **Question.** Can LLM-annotated data pre-train a small NER encoder?
+ **Method.** GPT-3.5 (`gpt-3.5-turbo-0301`) labels 1.35M C4 sentences; a filter keeps 1M. The result has 4.38M annotations and 200k concepts. RoBERTa-base gets contrastive pre-training on it (Section 3).
+ **Datasets.** Few-shot transfer on OntoNotes 5.0, BioNLP 2004, MIT Restaurant, MIT Movie (Section 4.1).
+ **Results.**
  + Token-level macro F1, average of 4 datasets (Figure 7 table): at k=1, RoBERTa 24.5, RoBERTa with NER-BERT data 32.3, NuNER 39.4. At k=64: 65.4, 67.6, 71.5.
  + Against UniNER-7B, entity-level micro F1 (Table 2): 8~16 shots 58.75 versus 57.89; 64~128 shots 70.30 versus 71.02.
  + Concept diversity and dataset size matter most; text diversity matters less (Section 5).
+ **Relevance.** LLM labels are a good pre-training signal for small encoders. Per-dataset MIT Movie numbers are in their Figure 13, which I did not read.

### B1.7 GLiNER2

+ **Citation.** Urchade Zaratiana, Gil Pasternak, Oliver Boyd, George Hurn-Maloney, Ash Lewis. arXiv 2507.18546.
+ **Question.** Can one GLiNER model do NER, classification and structured extraction?
+ **Method.** Trained on 254,334 examples. Real texts (135,698) and synthetic texts (118,636) are all annotated or generated by GPT-4o (Appendix B.1, Table 6).
+ **Results.** CrossNER zero-shot F1 (Table 3): GLiNER2 average 0.590, GPT-4o 0.599, GLiNER-M 0.615. Per domain, GLiNER2: AI 0.526, Literature 0.564, Music 0.632, Politics 0.679, Science 0.547.
+ **Note.** The text says GLiNER2 "achieves higher scores in AI (0.526 vs. 0.547)". Table 3 shows GPT-4o higher on AI. Trust the table.
+ **Relevance.** Our student's family is itself distilled from GPT-4o labels.

### B1.8 NoiseBench

+ **Citation.** Elena Merdjanovska, Ansar Aynetdinov, Alan Akbik. EMNLP 2024. arXiv 2405.07609.
+ **Question.** How does real label noise, including LLM noise, hurt NER compared with simulated noise?
+ **Method.** A CoNLL-03 subset with 6 noisy label sets. The clean reference is CleanCoNLL. The test split is the CoNLL-03 test split with CleanCoNLL labels (Section 2.1.1). The LLM set comes from GPT-3.5 through the Fabricator toolkit, static one-shot prompt (Section 2.1.6). FLERT (xlm-roberta-large) trains on each set.
+ **Results.**

  | Label set | Noise (100 minus entity F1) | Main error types | Clean-test F1 | F1 with matched simulated noise |
  |---|---|---|---|---|
  | Clean | 0 | none | 94.0 | not applicable |
  | Expert | 5.5 | wrong type 74.0% | 89.8 | 93.7 |
  | Crowd++ | 15.3 | missing 59.6% | 86.7 | 88.9 |
  | LLM (GPT-3.5) | 45.6 | non-entity (false positive) 45.4%, wrong type 28.3% | 62.6 | 68.6 |

  Source: Table 1 and Table 2 (mean of 3 runs).
  + Real noise costs about 2.5 F1 more than simulated noise on average; for the LLM set the gap is 6.0 (Table 2).
+ **Relevance.** Direct evidence on our CleanCoNLL labels: a weak LLM teacher gives very noisy NER labels. The noise is structured and hurts more than random noise. The paper does not study data selection.

## B2. Active learning with LLM annotators

### B2.1 LLMaAA: Making Large Language Models as Active Annotators

+ **Citation.** Ruoyu Zhang, Yanzeng Li, Yongliang Ma, Ming Zhou, Lei Zou. Findings of EMNLP 2023. arXiv 2310.19596.
+ **Question.** Can an LLM be the annotator inside an active learning loop for a small task model?
+ **Method.**
  + The teacher is ChatGPT with k-NN demonstrations from a 100-example gold set (Section 3).
  + The student is BERT with a linear tagger (NER) or classifier (RE).
  + Selection: random, maximum entropy, least confidence, k-means. Token scores are pooled by average or sum (Section 4.1).
  + Training uses automatic reweighting against the same 100 gold examples.
  + Budget: seed 50, then 50 per round for 9 rounds, 500 labels in total (Section 5.1).
+ **Datasets.** Chinese OntoNotes 4.0, English CoNLL03 (original labels), a Re-TACRED subset.
+ **Results (Table 1, F1, mean of 3 runs).**

  | Method | Chinese OntoNotes 4.0 | CoNLL03 | Re-TACRED subset |
  |---|---|---|---|
  | Prompting (teacher on test) | 70.73 | 81.33 | 73.77 |
  | Supervised on 100 gold | 73.00 | 77.94 | 74.28 |
  | LLMaAA, random selection | 70.21 | 79.17 | 76.41 |
  | LLMaAA, least confidence | 74.00 | 82.84 | 80.79 |

  + Uncertainty methods match random with only 30% to 40% of the data (Section 5.3.2, Figure 3).
  + Reweighting helps most where the teacher is weak (OntoNotes, Re-TACRED) and less on CoNLL03 (Section 5.3.3).
  + Without k-NN demos, teacher F1 drops by 21 points (OntoNotes) and 25 points (CoNLL) (Section 5.3.1).
  + Teacher strength (Table 4, Chinese OntoNotes, least confidence): GPT-3 teacher 29.49, student 56.63; ChatGPT 70.73, student 74.00; GPT-4 73.68, student 74.90. The student gain shrinks as the teacher gets stronger.
+ **Relevance.** The closest prior work. It shows uncertainty beats random with LLM labels on NER. But:
  + it is multi-round, not one round;
  + it uses gold demos and gold reweighting;
  + it never reports teacher accuracy on the selected sentences versus random sentences;
  + it reports the random arm only for ChatGPT, not for weak or strong teachers.

### B2.2 FreeAL: Towards Human-Free Active Learning in the Era of LLMs

+ **Citation.** Ruixuan Xiao, Yiwen Dong, Junbo Zhao, Runze Wu, Minmin Lin, Gang Chen, Haobo Wang. EMNLP 2023. arXiv 2311.15614.
+ **Question.** Can an LLM and a small model teach each other with no human labels?
+ **Method.**
  + GPT-3.5-Turbo labels the whole train set.
  + RoBERTa-base (BioMed-RoBERTa-base for biomedical data) trains with a small-loss rule: a 2-component GMM on per-sample loss splits clean from noisy samples (Section 4.2.1).
  + Clean samples become demos for the LLM, which relabels the noisy part. There are 4 rounds.
+ **Datasets.** SST-2, MR, SUBJ, TREC, CoNLL03, Medical Abstract, BC5CDR-Chemical, BC5CDR-Disease.
+ **Results (Table 3, test F1 for NER).**

  | Setting | CoNLL03 | BC5-Chemical | BC5-Disease |
  |---|---|---|---|
  | GPT-3.5 zero-shot ICL | 66.47 | 67.85 | 29.60 |
  | GPT-3.5 with FreeAL demos | 70.80 | 80.77 | 52.70 |
  | GPT-3.5 supervised ICL (standard retrieval) | 85.46 | 82.24 | 68.63 |
  | RoBERTa zero-shot distillation | 69.71 | 77.05 | 31.98 |
  | RoBERTa FreeAL | 76.12 | 81.13 | 58.90 |
  | RoBERTa supervised fine-tuning | 88.11 | 87.26 | 75.38 |

  + FreeAL with no human labels (94.66 SST-2, 90.20 MR) is near entropy AL with 50% human labels (94.29, 90.00) (Table 5).
  + Their BC5CDR split is 4,560 train and 4,797 test (Table 4). It differs from ours.
+ **Relevance.** FreeAL uses the idea that noisy LLM labels sit in the high-loss (hard) samples. It labels the whole pool, so it does not test budgeted selection.

### B2.3 LLMs in the Loop

+ **Citation.** Nataliia Kholodna, Sahib Julka, Mohammad Khodadadi, Muhammed Nurullah Gumus, Michael Granitzer. ECML PKDD 2024. arXiv 2404.02261.
+ **Question.** Can LLM labels replace human labels in active learning for low-resource NER?
+ **Method.** AfroXLMR-mini starts on 5% of the data. Each round adds the top 5% by mean token entropy, labelled by GPT-4-Turbo (Section 2).
+ **Datasets.** MasakhaNER 2.0 (Bambara, isiZulu in the AL runs).
+ **Results.**
  + Metric: entity-class accuracy, not entity F1 (Section 4).
  + Bambara: the full-data model scores 82%. AL with gold labels reaches it with 20% of the data and passes it at 30%. AL with GPT-4-Turbo labels reaches 76% and does not reach 82% within 5 rounds.
  + GPT-4-Turbo label accuracy is 84.5(9)% against gold.
  + Estimated cost is at least 42.45 times lower than human annotation.
+ **Relevance.** The same entropy selection hurts more with LLM labels than with gold labels. There is no random arm with LLM labels, so the uncertainty gain under LLM labels is not measured.

### B2.4 LLM on a Budget: Active Knowledge Distillation

+ **Citation.** Viviana Luccioli, Rithika Iyengar, Ryan Panley, et al. (Federal Reserve Board). arXiv 2511.11574.
+ **Question.** Can uncertainty sampling cut the number of LLM teacher calls?
+ **Method.** M-RARU, a randomized accept/reject uncertainty sampler. The teacher is a local `gemma-3-27b-it-qat-q4_0`. Students are SVM, LDA, RF, GBDT, DistilBERT.
+ **Datasets.** 125,179 public comments to the Federal Reserve (5 classes); 12,288 news headlines (GDP rising, falling, flat).
+ **Results.** Up to 80% fewer samples than random sampling for the same accuracy (abstract).
+ **Relevance.** The problem setup treats the LLM labels as the true labels (Section on the problem). So teacher error is invisible by design. Classification only.

### B2.5 LLKD: Knowledge Distillation from LLMs via Unlabeled Data

+ **Citation.** Juanhui Li, Sreyashi Nag, Hui Liu, et al. arXiv 2411.08028 (v3).
+ **Question.** Which LLM-labelled samples should a student train on?
+ **Method.** At each step, keep samples where the teacher is confident and the student is uncertain. The teacher is LLaMA (Gemma in one ablation); the student is RoBERTa.
+ **Datasets.** PubMed-RCT-20k, Yahoo! Answers, Emotions, Arxiv-10, BiosBias.
+ **Results.** On PubMed-RCT-20k: 5.82% relative F1 gain over the best baseline using 3.7% of training samples (Section 5, Tables 1 and 2). Most datasets use under 25% of samples.
+ **Relevance.** The design assumes that student-uncertain samples help and that teacher-unsure samples hurt. It filters after the teacher labels everything. It does not measure teacher accuracy on the student-uncertain set. Classification only.

## B3. Uncertainty sampling for NER and active learning background

### B3.1 Deep Active Learning for Named Entity Recognition

+ **Citation.** Yanyao Shen, Hyokun Yun, Zachary C. Lipton, Yakov Kronrod, Animashree Anandkumar. ICLR 2018. arXiv 1707.05928.
+ **Question.** Can deep NER models learn well from few actively chosen sentences?
+ **Method.** A CNN-CNN-LSTM tagger. Selection by least confidence (LC), MNLP, or BALD, against random.
+ **The MNLP score.** LC favours long sentences. MNLP divides the log-probability of the best tag sequence by sentence length n: `max over y of (1/n) * sum_i log P(y_i | ...)`. Sentences with the lowest value are chosen (Section 4).
+ **Datasets.** CoNLL-2003 English (model check), OntoNotes 5.0 English and Chinese (AL runs).
+ **Results.**
  + All AL methods beat random clearly. MNLP and BALD beat LC slightly in early rounds (Section 5.2, Figure 4).
  + With 24.9% of the English data, AL reaches 99% of the best full-data deep model F1 (86.86). Chinese needs 30.1% (full-data 75.63).
  + Their tagger scores 90.69 ± 0.19 F1 on CoNLL-2003 test (Table 3).
+ **Relevance.** The classic evidence that uncertainty beats random for NER, with gold labels. Our MNLP-style normalisation idea comes from here.

### B3.2 A Survey of Active Learning for NLP

+ **Citation.** Zhisong Zhang, Emma Strubell, Eduard Hovy. EMNLP 2022. arXiv 2210.10109.
+ **Points for us.**
  + Informativeness alone risks sampling bias and outliers (Section 2.2).
  + Noisy crowd labels can reduce the value of AL (Section 5, "Crowdsourcing and Noise", citing Rehbein and Ruppenhofer 2011).
  + Cold start: random seed selection is the most common choice (Section 4).

### B3.3 Scoping Review of AL Strategies for Entity Recognition

+ **Citation.** Philipp Kohl, Yoka Krämer, Claudia Fohry, Bodo Kraft. arXiv 2407.03895.
+ **Results.** 62 papers, 106 strategies: 60 exploitation (60% of them uncertainty-based), 14 exploration, 32 hybrid. All use F1. Only 6 papers report hardware and 13 report timing. 57 datasets, 26 public (abstract, Section 4).
+ **Relevance.** Supports our choice to log GPU time and wall time.

### B3.4 Cold-start Active Learning through Self-supervised Language Modeling

+ **Citation.** Michelle Yuan, Hsuan-Tien Lin, Jordan Boyd-Graber. EMNLP 2020. arXiv 2010.09535.
+ **Method.** ALPS uses the masked language model loss as a proxy for uncertainty, so a cold model can still select.
+ **Result.** Higher accuracy in fewer rounds than baselines on four text classification datasets (abstract).
+ **Relevance.** Our GLiNER2.5 student is not cold (it is zero-shot capable). This paper shows why a cold student's uncertainty is unreliable.

### B3.5 Mind Your Outliers (abstract only)

+ **Citation.** Siddharth Karamcheti, Ranjay Krishna, Li Fei-Fei, Christopher D. Manning. ACL-IJCNLP 2021. arXiv 2107.02331.
+ **Result from the abstract.** On visual question answering, many AL methods fail to beat random. The cause is "collective outliers": examples AL prefers but models cannot learn. Removing them raises AL efficiency.
+ **Relevance.** Uncertain examples can be hard for everyone. With an LLM teacher, they may also be the ones the teacher gets wrong.

### B3.6 Optimal Labeler Assignment and Sampling with Imperfect Labels

+ **Citation.** Pouya Ahadi, Blair Winograd, Camille Zaug, Karunesh Arora, Lijun Wang, Kamran Paynabar. arXiv 2512.12870 (preprint under review).
+ **Point.** The paper assumes that uncertain samples get more wrong labels, citing Du and Ling (2010) (Section 1). Its labelers are simulated, and its data are tabular (Heart, Ionosphere, Sonar, Spambase).
+ **Relevance.** The assumption "noise grows with uncertainty" is standard, but here it is not measured with LLM labelers.

---

# What is already known and what is open

## Known

1. **LLM labels can train students that beat the teacher.** Wang et al. 2021, UniNER, LLMaAA, LLKD all show it.
2. **LLM labels for NER are noisy, and the noise is structured.** GPT-3.5 labels on CleanCoNLL text have 45.6% entity-level noise, mostly false positives and wrong types (NoiseBench Table 1). Real noise hurts more than matched random noise (6.0 F1 for the LLM set, Table 2).
3. **Teacher confidence tracks teacher accuracy.** Low-confidence GPT-3 labels are much less accurate (Wang et al. 2021, Figure 5).
4. **With gold labels, uncertainty beats random for NER.** Shen et al. 2018.
5. **With LLM labels, uncertainty beat random in one multi-round NER setup.** LLMaAA: CoNLL03 82.84 versus 79.17. That setup had gold demos, gold reweighting and a ChatGPT teacher.
6. **With LLM labels, uncertainty sampling falls short of gold-label AL.** LLMs in the Loop: 76% versus the 82% full-data score.
7. **Several methods already assume that hard samples carry teacher noise.** FreeAL removes high-loss samples. LLKD keeps only teacher-confident samples. Ahadi et al. model noise that grows with uncertainty.

## Open (the gap our paper fills)

No paper I read does all of the following at once:

1. Measure the **teacher error rate on the student-uncertain set** and on a **random set from the same frozen pool**.
2. Link that error gap to the **downstream gain** of uncertainty over random.
3. Do it in **one round**: one selection, one teacher call per sentence, no feedback loop.
4. Vary the **teacher strength** (weak and strong, local and API) while the pool, seeds and budget stay fixed.
5. Cover more than one structure: NER, relations, multi-label classification and slots.
6. Use a **GLiNER-family student** and keep sentences with no gold entity in the pool.

In simple words: prior work shows that "pick the hard sentences" helps with human labels. It also shows that LLMs make more mistakes on hard cases. Nobody has measured where these two effects cancel.

Example: the student picks 500 sentences it is unsure about. The teacher labels them.

+ If the teacher is 90% right on these sentences, the student gains from the hard sentences.
+ If the teacher is only 55% right on them, the student learns the mistakes. Then 500 random sentences with 85% teacher accuracy can win.

```
  frozen pool
      |
      +--> random 500 -------------> teacher labels --> error e_rand --> student F1_rand
      |
      +--> most uncertain 500 -----> teacher labels --> error e_unc  --> student F1_unc
                                                            |
                                   benefit = F1_unc - F1_rand
                                   question: as (e_unc - e_rand) grows,
                                   when does the benefit drop to zero or below?
```

LLMaAA Table 4 hints at the teacher effect: the student gain over the teacher shrinks from +27.1 F1 (GPT-3) to +1.2 F1 (GPT-4). But it has no random arm per teacher, so it cannot separate selection from teacher strength.

---

# Part C: reference scores for our datasets

Read the "Setting" and "Labels" columns before you compare. "Orig." means original labels, not our version.

## C1. CleanCoNLL and CoNLL-03 (4 types)

| Source | Model | Setting | Labels | Metric | Score |
|---|---|---|---|---|---|
| CleanCoNLL Table 4 | Biaffine (xlm-roberta-large) | fine-tuned, full train | CleanCoNLL | test F1, 3 seeds | 97.08 |
| CleanCoNLL Table 4 | FLERT (xlm-roberta-large) | fine-tuned, full train | CleanCoNLL | test F1, 3 seeds | 96.98 |
| CleanCoNLL Table 4 | FLERT | fine-tuned, full train | CoNLL-03 orig. | test F1 | 93.94 |
| NoiseBench Table 2 | FLERT | fine-tuned, NoiseBench subset, clean labels | CleanCoNLL test | entity F1, 3 runs | 94.0 |
| NoiseBench Table 2 | FLERT | fine-tuned on GPT-3.5 labels (45.6% noise) | CleanCoNLL test | entity F1, 3 runs | 62.6 |
| GLiNER Table 2 | ChatGPT | zero-shot (numbers from UniNER) | CoNLL-03 orig. | F1 | 52.5 |
| GLiNER Table 2 | UniNER-7B | zero-shot | CoNLL-03 orig. | F1 | 72.2 |
| GLiNER Table 2 | GLiNER-L | zero-shot | CoNLL-03 orig. | F1 | 64.6 |
| GLiNER Table 4 | GLiNER-L (with Pile-NER pre-training) | supervised on 20-dataset mix | CoNLL-03 orig. | F1 | 92.6 |
| UniNER Table 2 | UniNER-7B | supervised on 20-dataset mix | CoNLL-03 orig. | F1 | 93.30 |
| FreeAL Table 3 | GPT-3.5-Turbo | zero-shot ICL | CoNLL-03 orig. | test F1, 3 runs | 66.47 |
| FreeAL Table 3 | RoBERTa-base | supervised fine-tuning | CoNLL-03 orig. | test F1 | 88.11 |
| LLMaAA Table 1 | ChatGPT | k-NN demos from 100 gold | CoNLL-03 orig. | test F1 | 81.33 |
| LLMaAA Table 1 | BERT, 500 ChatGPT labels, least confidence | multi-round AL | CoNLL-03 orig. | test F1, 3 runs | 82.84 |
| Shen et al. Table 3 | CNN-CNN-LSTM | fine-tuned, full train | CoNLL-03 orig. | test F1 | 90.69 |
| GLiNER2 paper | GLiNER2 | any | any | any | not reported |

## C2. BC5CDR

| Source | Model | Setting | Types | Metric | Score |
|---|---|---|---|---|---|
| BLURB Table 6 | PubMedBERT | fine-tuned | chemical only | entity F1 | 93.33 |
| BLURB Table 6 | PubMedBERT | fine-tuned | disease only | entity F1 | 85.62 |
| BLURB Table 6 | BioBERT | fine-tuned | chemical / disease | entity F1 | 92.85 / 84.70 |
| UniNER Table 2 | BERT-base | supervised on 20-dataset mix | both | F1 | 85.28 |
| UniNER Table 2 | UniNER-7B | supervised on 20-dataset mix | both | F1 | 89.34 |
| GLiNER Table 4 | GLiNER-L (with / without Pile-NER) | supervised on 20-dataset mix | both | F1 | 88.7 / 88.7 |
| GLiNER Table 2 | ChatGPT | zero-shot | both | F1 | 52.4 |
| GLiNER Table 2 | UniNER-7B | zero-shot | both | F1 | 68.0 |
| GLiNER Table 2 | GLiNER-L | zero-shot | both | F1 | 66.4 |
| FreeAL Table 3 | GPT-3.5-Turbo | zero-shot ICL | chemical / disease | test F1 | 67.85 / 29.60 |
| FreeAL Table 3 | BioMed-RoBERTa-base | supervised fine-tuning | chemical / disease | test F1 | 87.26 / 75.38 |
| FreeAL Table 3 | BioMed-RoBERTa-base | FreeAL, no human labels | chemical / disease | test F1 | 81.13 / 58.90 |
| Li et al. 2016 | best BioCreative V system | DNER task (with normalisation) | disease | F-score | 86.46 |
| Li et al. 2016 Table 3 | human annotators | inter-annotator agreement | disease / chemical | Jaccard | 0.8749 / 0.9605 |

Note: FreeAL's zero-shot disease F1 (29.60) is far below chemical (67.85). A zero-shot LLM teacher on the disease type is weak in this setting.

## C3. MIT Movie (all on original MIT labels)

| Source | Model | Setting | Metric | Score |
|---|---|---|---|---|
| Liu et al. 2013 Table 4 | semi-Markov CRF, best features | supervised, own 80/20 split of 12,000 sentences | F-score | 88.58 |
| UniNER Table 2 | BERT-base | supervised on 20-dataset mix | F1 | 88.78 |
| UniNER Table 2 | InstructUIE 11B (re-evaluated) | supervised | F1 | 89.58 |
| UniNER Table 2 | UniNER-7B | supervised | F1 | 90.17 |
| GLiNER Table 4 | GLiNER-L (with / without Pile-NER) | supervised on 20-dataset mix | F1 | 87.9 / 87.5 |
| GLiNER Table 1 | ChatGPT | zero-shot | F1 | 5.3 |
| GLiNER Table 1 | Vicuna-7B / Vicuna-13B | zero-shot | F1 | 6.0 / 0.9 |
| GLiNER Table 1 | InstructUIE 11B | zero-shot | F1 | 63.0 |
| GLiNER Table 1 | GoLLIE 7B | zero-shot | F1 | 63.0 |
| GLiNER Table 1 | UniNER-7B / UniNER-13B | zero-shot | F1 | 42.4 / 48.7 |
| GLiNER Table 1 | GLiNER-S / M / L | zero-shot | F1 | 46.9 / 42.9 / 57.2 |
| UniNER Table 5 | ChatGPT / UniNER-7B | zero-shot, partial match counts 0.5 | F1 | 5.9 / 46.9 |
| UniNER Table 3 | UniNER-7B (instruction-tuned plus supervised on other datasets) | out-of-domain supervised | F1 | 61.2 |
| GLiNER2 paper | GLiNER2 | any | any | not reported |

Note: ChatGPT scores 5.3 F1 zero-shot on MIT Movie, far below its other OOD scores (GLiNER Table 1). The source papers do not explain why. Do not treat this number as a guide for a modern teacher.

## C4. CrossRE

Sizes only, see A4. Scores live on the relations page.

---

## Sources

+ CleanCoNLL: https://arxiv.org/abs/2310.16225
+ BC5CDR: https://doi.org/10.1093/database/baw068 (read via Europe PMC, PMC4860626)
+ BLURB / PubMedBERT: https://arxiv.org/abs/2007.15779
+ MIT Movie (Asgard): https://sls.csail.mit.edu/publications/2013/Liu_ICASSP-2013.pdf ; data at https://groups.csail.mit.edu/sls/downloads/movie/
+ CrossRE: https://arxiv.org/abs/2210.09345
+ GLiNER2.5 Multi model card: https://huggingface.co/fastino/gliner2.5-multi-v1
+ Wang et al. 2021: https://arxiv.org/abs/2108.13487
+ Ding et al. 2023: https://arxiv.org/abs/2212.10450
+ Gilardi et al. 2023: https://arxiv.org/abs/2303.15056
+ UniNER: https://arxiv.org/abs/2308.03279
+ GLiNER: https://arxiv.org/abs/2311.08526
+ NuNER: https://arxiv.org/abs/2402.15343
+ GLiNER2: https://arxiv.org/abs/2507.18546
+ NoiseBench: https://arxiv.org/abs/2405.07609
+ LLMaAA: https://arxiv.org/abs/2310.19596
+ FreeAL: https://arxiv.org/abs/2311.15614
+ LLMs in the Loop: https://arxiv.org/abs/2404.02261
+ LLM on a Budget: https://arxiv.org/abs/2511.11574
+ LLKD: https://arxiv.org/abs/2411.08028
+ Shen et al.: https://arxiv.org/abs/1707.05928
+ Zhang, Strubell, Hovy survey: https://arxiv.org/abs/2210.10109
+ Kohl et al. scoping review: https://arxiv.org/abs/2407.03895
+ Yuan et al. cold start: https://arxiv.org/abs/2010.09535
+ Karamcheti et al. (abstract only): https://arxiv.org/abs/2107.02331
+ Ahadi et al.: https://arxiv.org/abs/2512.12870

Related page: `docs/research/2026-09-25-1050-related-work-scan.md` (abstract-level scan of more 2024-2026 papers).

## Changelog
- 2026-09-25 23:25 CEST - Created from a full read of the papers.
