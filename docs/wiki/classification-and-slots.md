---
title: GLiNER-style classification and slot filling
date: 2026-09-25 23:25 CEST
author: Claude (research agent)
type: wiki
status: draft
---

# GLiNER-style classification and slot filling

This page collects what the literature says about two of our four tasks:

1. Multi-label text classification (Hallmarks of Cancer).
2. Slot filling (MASSIVE en-US and fr-FR).

Every number on this page comes from a paper, a model card, or source code that I read on 25 Sep 2026. The page names the table or file for each number. "Not reported" means that the source does not give the fact.

## 1. GLiClass (Knowledgator)

**Citation.** Stepanov, Shtopko, Vodianytskyi, Lukashov, Yavorskyi, Yaroshenko. "GLiClass: Generalist Lightweight Model for Sequence Classification Tasks." arXiv 2508.07662v1, 11 Aug 2025. https://arxiv.org/abs/2508.07662 . Code: https://github.com/Knowledgator/GLiClass . Model card: https://huggingface.co/knowledgator/gliclass-large-v3.0 .

### Method (section 2.1)

+ Uni-encoder - text and labels go through one encoder together. Each label gets a `<<LABEL>>` prefix token and sits next to the text (2.1.2).
+ Joint attention - the encoder sees label-label, text-label and label-text interactions. Pairwise cross-encoders cannot see other labels (2.1.3).
+ Pooling - first-token, mean, or attention-weighted pooling for text and label vectors (2.1.4).
+ Scorer - dot product divided by a temperature, or a small MLP on the concatenated text and label vectors (2.1.5, eq. 1 and 2).
+ Layer re-weighting - a squeeze-excitation block mixes all encoder layers (2.1.6).
+ Token-level contrastive loss - an extra loss that trains each token to find itself in its own sequence (2.1.7, eq. 8).
+ Variants - uni-encoder, bi-encoder, fused bi-encoder, encoder-decoder (2.1.8).
+ Backbones - DeBERTa-v3 and ModernBERT. The paper says DeBERTa models "consistently outperform" ModernBERT models (2.1).

### Loss

+ Supervised pipeline - focal loss (2.3.1). The paper does not print the supervised focal formula.
+ RL pipeline - a PPO loss plus value, KL and entropy terms (eq. 25 to 29). A focal factor can multiply the PPO loss (eq. 30).
+ Default RL settings - clip 0.2, 3 RL iterations. Entropy, KL, focal alpha, focal gamma and label smoothing are all "-1 (disabled)" by default (2.3.2).
+ Post-training focal alpha - 0.7 for all v3.0 models (Table 1). Focal gamma for post-training: not reported.
+ Code default - `focal_loss_with_logits` in `gliclass/loss_functions.py` is the RetinaNet focal loss with `alpha=0.25`, `gamma=2` as function defaults. The docstring says alpha "balance[s] positive vs negative examples" and gamma "balance[s] easy vs hard examples". The `train.py` command-line defaults are `--focal_loss_alpha -1` and `--focal_loss_gamma -1`, so focal loss is off unless you set it.

### Training data and recipe (sections 2.2 and 2.3)

| Stage | Data | Method |
|---|---|---|
| Pre-training | 1.2M examples: text classification, sentiment, NLI | supervised; learns the class tokens |
| Mid-training | a subset of the pre-training corpus | RL (PPO) trainer; "modest but consistent gains in macro F1" |
| Post-training | logic/NLI stream (CommonsenseQA plus 2,000 synthetic logic examples), pattern-focused stream (2,000 texts in length buckets from 0 to 1024 words, 50 true and 50 false GPT-4o labels per text), MultiNLI | LoRA |

+ Split - 90% train, 10% test from the JSON data. Max length 1024 tokens (2.3.1).
+ Learning rates - encoder 1e-5, classifier layers 3e-5, weight decay 0.01 for both (2.3.2).
+ LoRA - very high ranks with alpha = 2 x rank (Table 1):

| Model | LoRA r | LoRA alpha | Focal alpha |
|---|---|---|---|
| gliclass-edge-v3.0 | 1536 | 3072 | 0.7 |
| gliclass-modern-base-v3.0 | 512 | 1024 | 0.7 |
| gliclass-modern-large-v3.0 | 768 | 1536 | 0.7 |
| gliclass-base-v3.0 | 384 | 768 | 0.7 |
| gliclass-large-v3.0 | 384 | 768 | 0.7 |

+ Rank finding - the edge model "trained more stably when using higher-rank (over-parameterized) LoRA adapters" (2.3.3).
+ Pattern stream - the authors duplicate each text and change the number of positive and negative labels in the copy "to diversify label density" (2.2).

`train.py` defaults (repo, not the paper): 3 epochs, batch size 8, warmup ratio 0.05, dropout 0.3, `shuffle_labels=True`, and label augmentation on by default:

+ Random label removal - 0.05.
+ Random label addition - 0.05.
+ Random text addition - 0.05.
+ Random description added - 0.05.
+ Random synonyms - 0.05.
+ Random examples added - 0.1, at most 5.

### Results

Zero-shot F1 (Table 3, selected rows; the table does not say micro or macro):

| Dataset | large-v3.0 | base-v3.0 | modern-large | modern-base | edge |
|---|---|---|---|---|---|
| sst2 | 0.9192 | 0.8959 | 0.9330 | 0.8959 | 0.8199 |
| 20_news_groups | 0.5958 | 0.4759 | 0.3905 | 0.3433 | 0.2217 |
| massive (intent) | 0.5649 | 0.5040 | 0.3905 | 0.3442 | 0.2414 |
| banking | 0.5574 | 0.4698 | 0.3683 | 0.3561 | 0.0272 |
| snips | 0.9692 | 0.9474 | 0.7707 | 0.5663 | 0.5257 |
| AVERAGE (14 sets) | 0.7193 | 0.6764 | 0.6197 | 0.5577 | 0.4900 |

+ Cross-encoders - the best, deberta-v3-large-zeroshot-v2.0, averages 0.6821 (Table 4). gliclass-large-v3.0 is +0.037 above it.
+ Model card mismatch - the model card's top table gives other averages (large 0.7001, base 0.6556, edge 0.4873). Its per-dataset table repeats the paper (0.7193). Cite the paper numbers.
+ MASSIVE row - this is intent classification, not slots. The locale is not reported.

Few-shot, 8 examples per label (Table 5, average of 8 datasets):

| Model | 0-shot | 8-shot | Gain |
|---|---|---|---|
| edge-v3.0 | 0.3774 | 0.5662 | +0.1888 |
| modern-base-v3.0 | 0.4445 | 0.6539 | +0.2094 |
| modern-large-v3.0 | 0.5205 | 0.7082 | +0.1877 |
| base-v3.0 | 0.5804 | 0.6871 | +0.1067 |
| large-v3.0 | 0.6202 | 0.7265 | +0.1063 |

+ MASSIVE intent, base-v3.0 - 0.5041 at 0-shot, 0.7008 at 8-shot (Table 5).
+ Few-shot training settings (epochs, learning rate, LoRA or full) - not reported.

### Label count effects

+ Speed - throughput falls 7% to 20% from 1 to 128 labels. A cross-encoder falls about 52 times (Table 6).
+ Accuracy - "performance can degrade with larger label sets, as seen in datasets like banking77" (section 4).
+ Attention - "as the number of labels increases, attention between tokens of labels and label tokens ... diminishes", and "under extreme label-to-text token ratios (many labels and short texts), text representations degrade" (2.3.3).
+ Context limit - about 1024 tokens. 1000+ labels need truncation or batching (section 4).

### Limitations (sections 4 and 5)

+ "calibration variability across datasets".
+ "sensitivity to extreme label-to-text ratios".
+ "variability on fine-grained taxonomies".
+ No multilingual model yet. All evaluation sets are English.
+ No multi-label benchmark with a gold label set is reported. All tables are single-label style datasets.

## 2. GLiNER2 (Fastino)

**Citation.** Zaratiana, Pasternak, Boyd, Hurn-Maloney, Lewis. "GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface." arXiv 2507.18546v1, 24 Jul 2025. https://arxiv.org/abs/2507.18546 . Code and tutorials: https://github.com/fastino-ai/GLiNER2 (folder `tutorial/`).

### Method (section 2, Appendix A)

+ Input - `[Task Prompt] [SEP] [Input Text]`. Learned special tokens `[P]`, `[E]`, `[C]`, `[L]`, `[SEP]` are randomly initialised.
+ Classification - `[P] task ([L] l1 [L] l2 ...) [SEP] x1 ... xN`. Each `[L]` token gives a label vector. An MLP maps it to one logit (Appendix A, eq. 2).
+ Single-label vs multi-label - softmax over the logits for single-label, sigmoid per logit for multi-label (Appendix A).
+ Structure (JSON) - `[P] parent ([C] a1 [C] a2 ...)`. An MLP on `[P]` predicts the instance count K as a 20-class problem (counts 0 to 19). Count embeddings then make K copies of each `[C]` field. Each copy scores text spans like NER (Appendix A).
+ Span threshold - "Spans with a predicted probability above 0.5 for any entity type are selected" (Appendix A).
+ Label descriptions - supported for classification and entities (Table 1, section 2).

### Loss

+ The paper does not state the classification or structure loss.
+ The installed `gliner2` 2.0.0 code (our `.venv`) uses plain binary cross-entropy with logits for classification, `reduction="sum"`. See `models/span/model.py` line 385 and `models/boundary/model.py` `_classification_loss`.
+ The boundary model divides the sum by the number of supervised labels and multiplies it by `classification_loss_weight` (default 1.0).
+ No class weight, no positive weight, no focal term for classification. The code applies BCE also to single-label tasks. Softmax is only an inference choice.
+ The boundary span head has an option `boundary_marginal_loss = "asymmetric_focal"` (default `"bce"`, gamma positive 0.0, gamma negative 2.0, clip 0.05) in `configuration.py`. This option is for spans, not for classification.

### Training recipe (Appendix B, Table 5)

| Setting | Value |
|---|---|
| Epochs | 5 |
| Optimizer | AdamW |
| LR backbone | 1e-5 |
| LR task layers | 2e-5 |
| Weight decay | 0.01 |
| Warmup | 1,000 steps, linear |
| Gradient clipping | 1.0 |

### Training data (section 3.1, Appendix B.1, Table 6)

+ Total - 254,334 examples.
+ Real text - 135,698 documents (law 19,798; PubMed 16,400; Wikipedia 17,909; arXiv 7,135; news 74,456). GPT-4o made all labels.
+ Synthetic - 118,636 GPT-4o examples (emails, messages, resumes, e-commerce, banking, sports).

### Results

Zero-shot classification, Table 2 (the text calls the scores "accuracy"):

| Dataset | Labels | GPT-4o | GLiClass (base-v1.0) | DeBERTa-v3 | GLiNER2 |
|---|---|---|---|---|---|
| SNIPS | 7 | 0.97 | 0.80 | 0.77 | 0.83 |
| Banking77 | 77 | 0.78 | 0.21 | 0.42 | 0.70 |
| Amazon Intent (MASSIVE) | 31 | 0.72 | 0.51 | 0.59 | 0.53 |
| SST-2 | 2 | 0.94 | 0.90 | 0.92 | 0.86 |
| IMDB | 2 | 0.95 | 0.92 | 0.89 | 0.87 |
| AG News | 4 | 0.85 | 0.68 | 0.68 | 0.74 |
| 20 Newsgroups | 20 | 0.68 | 0.36 | 0.54 | 0.49 |
| Average | | 0.84 | 0.63 | 0.69 | 0.72 |

+ The Amazon Intent row cites FitzGerald et al. (2022), the MASSIVE paper. The table gives 31 labels, not 60. The paper does not explain the 31.
+ The GLiClass baseline is the old `gliclass-base-v1.0`, not v3.0 (Appendix B).
+ CPU latency with 50 labels - GLiNER2 208 ms, DeBERTa cross-encoder 16,897 ms (Table 4).
+ Structure extraction - "was not evaluated due to the absence of established zero-shot benchmarks" (3.2). So no JSON or slot score exists in the paper.
+ Fine-tuned classification - not reported. Multi-label classification benchmark - not reported.

### Tutorial advice (repo `tutorial/`)

`1-classification.md`:

+ "Adding descriptions significantly improves accuracy by providing context." No numbers.
+ "Lower thresholds for multi-label (0.3-0.5), higher for single-label (0.5-0.7)."
+ Example multi-label calls use `cls_threshold` from 0.3 to 0.4. `class_act` can force `"sigmoid"`, `"softmax"` or `"auto"`.
+ The setup names `fastino/gliner2.5-multi-v1` as the multilingual checkpoint.

`14-constrained_classification.md`:

+ `Classifier` with `.multi(name, labels, min_labels=0)` allows an empty label set. `.single` forces exactly one label.
+ Task options include `threshold`, `activation`, `temperature`.

`3-json_extraction.md`:

+ Field format `name::type::description` and `name::[a|b|c]::type::description`.
+ Advice: "Add descriptions for complex or domain-specific fields".

`8-train_data.md`:

+ Classification JSONL supports `label_descriptions`, `prompt` and few-shot `examples` per task.
+ Structures allow empty values (`""`) for absent fields.

`9-training.md`:

+ LoRA defaults - `r=16`, `alpha=32`, `dropout=0.1`, `task_lr=5e-4`. When LoRA is on, `task_lr` trains both the adapters and the task heads.
+ Default LoRA targets - all modules: `encoder`, `span_rep`, `classifier`, `count_embed`, `count_pred`.
+ "LoRA performance worse than full fine-tuning" - increase rank to 32, add task heads, raise `task_lr` to 1e-3, train longer (up to 20 epochs).
+ Full fine-tuning LRs - encoder 1e-6 to 5e-5 (typically 1e-5), task 1e-4 to 1e-3 (typically 5e-4).

### GLiNER2.5 model card

https://huggingface.co/fastino/gliner2.5-multi-v1 : mDeBERTa-v3-base encoder, 287M parameters, boundary architecture. The card shows no benchmark scores for classification or structures.

### Limitations

+ No reported score for multi-label classification, fine-tuned classification, or structure/slot extraction.
+ Zero-shot classification averages 0.72 against 0.84 for GPT-4o (Table 2).
+ The paper trains on GPT-4o labels only. It reports no study of label noise.

## 3. Other GLiNER-family work on classification or slots

+ GLiNER multi-task (Stepanov and Shtopko, arXiv 2406.12925) - evaluates NER, QA, summarization and relation extraction. It reports no text classification scores (Tables 2 to 6). GLiClass section 1 says classification in that model exposed "limitations that highlighted the need for a more specialized solution".
+ Rana, Hacioglu, Gopalan, Boothalingam, "Zero-shot Slot Filling in the Age of LLMs for Dialogue Systems," arXiv 2411.18980 (Uniphore). A GLiNER model extracts slot values in a call-centre pipeline, then a fine-tuned LLM gets a reduced slot list. They write that "GLiNER excels in recall for extracting slot values but lacks precision and cannot extract abstractive slots". Relative F1 gain over their legacy system on unseen data: only GLiNER 2%, GLiNER plus constraints 16%, GLiNER plus fine-tuned LLM plus constraints 34% (Table 8). Internal data only. No MASSIVE or SNIPS score.
+ I found no paper that fine-tunes a GLiNER-style model on MASSIVE slots or on Hallmarks of Cancer. Searches: arXiv and web for "GLiNER slot filling", "GLiNER classification", "zero-shot text classification encoder label tokens". Semantic Scholar search: not done.

## 4. Hallmarks of Cancer corpus

### Baker et al. 2016 (the corpus paper)

**Citation.** Baker, Silins, Guo, Ali, Högberg, Stenius, Korhonen. "Automatic semantic classification of scientific literature according to the hallmarks of cancer." Bioinformatics 32(3):432-440, 2016. https://doi.org/10.1093/bioinformatics/btv585 . I read the Oxford Academic page through a web tool (full text), not the PDF.

+ Size - 1,499 PubMed abstracts.
+ Annotation - one expert with 15+ years in cancer research. Kappa 0.81 on 155 abstracts.
+ Sentence level - "only sentences describing findings or conclusions of the study in question were included". The abstract label set is the union of its sentence labels.
+ Multi-label - 40% of abstracts have more than one label (60.2% one, 28.5% two, 8.9% three).
+ Per-label counts, abstracts / sentences (Table 2): proliferative signaling 462 / 993, growth suppressors 242 / 468, cell death 430 / 883, replicative immortality 115 / 295, angiogenesis 143 / 357, invasion and metastasis 291 / 667, genome instability 333 / 771, inflammation 194 / 437, cellular energetics 105 / 213, immune destruction 108 / 226.
+ Model - SVM with RBF kernel on bag-of-words, noun bigrams, grammatical relations, verb classes, named entities, MeSH terms and chemical lists.
+ Evaluation - abstract level, 4-fold cross-validation.
+ Result - average F 76.9%, per label from 65.8 (growth suppressors) to 90.9 (cellular energetics) (Table 4).

### Baker, Korhonen, Pyysalo 2016 (CNN, BioNLP workshop)

**Citation.** "Cancer Hallmark Text Classification Using Convolutional Neural Networks." BioNLP 2016, https://aclanthology.org/W16-5101.pdf .

+ Task - ten binary abstract-level tasks, one per hallmark. 70/10/20 random split of 1,852 abstracts (Table 1).
+ Imbalance - "negative examples outnumbering positives more than 10-fold for a number of the labels". Example: cellular energetics 74 positive vs 1,229 negative in train (Table 1).
+ Oversampling positives to match negatives raised dev F from 85.3% to 86.1% and dev AUC from 97.3% to 97.5% (4.1).
+ Test average F (Table 4) - SVM bag-of-words 69.2, SVM rich features 76.8, CNN base 76.6, CNN tuned 81.0.
+ Threshold note - the rich SVM beats the base CNN on F, but the CNN wins on AUC. The authors suggest the SVM advantage "may be due in part to a better position of the decision boundary" (4.2).

### BLUE (Peng et al. 2019)

**Citation.** "Transfer Learning in Biomedical NLP: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets." arXiv 1906.05474.

+ Data - 1,580 abstracts, split 1,108 / 157 / 315 (Table 1). BLURB says this version dropped the 272 abstracts with no hallmark.
+ Metric - "example-based F1-score on the abstract level" (3.4).
+ Test F1 (Table 3) - ELMo 80.0, BioBERT 82.9, BERT-Base (PubMed) 85.3, BERT-Large (PubMed) 87.3, previous SOTA 81.5.

### BLURB (Gu et al. 2020)

**Citation.** Gu, Tinn, Cheng, Lucas, Usuyama, Liu, Naumann, Gao, Poon. "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing." arXiv 2007.15779.

+ Data - the expanded 1,852-abstract release, including abstracts with no hallmark. Split 1,295 / 186 / 371 abstracts (Table 3). BLURB made the split (2.3.5).
+ Labels - ten top-level hallmarks (the corpus has 37 fine-grained classes).
+ Level - abstract level, "follow the common practice" (2.3.5).
+ Metric - **micro F1 across the ten cancer hallmarks** (2.3.5, Table 3).
+ Model - [CLS] vector plus a linear layer (2.4).
+ Fine-tuning - Adam, slanted triangular schedule (10% warmup), dropout 0.1. Dev grid: LR {1e-5, 3e-5, 5e-5}, batch {16, 32}, epochs 2 to 60. Mean of five runs (2.5).

HoC test micro-F1 (Table 6):

| Model | HoC |
|---|---|
| BERT uncased | 80.20 |
| BERT cased | 80.12 |
| RoBERTa | 79.66 |
| BioBERT | 81.54 |
| SciBERT uncased | 80.66 |
| SciBERT cased | 81.16 |
| ClinicalBERT | 80.74 |
| BlueBERT | 80.48 |
| PubMedBERT | 82.32 |

Other PubMedBERT variants: 81.76 and 81.74 (vocabulary and WWM ablations, Table 7), 83.14 (Wiki+Books then PubMed, Table 9), 82.62 (PubMed + PMC, longer training, Table 10).

### Our copy

`bigbio/hallmarks_of_cancer` reads the upstream GitHub release and the BLURB PMID split files. See `docs/research/2026-09-25-1230-dataset-sources.md` section 5.

+ Same documents - 1,295 / 186 / 371 PMIDs, the BLURB split.
+ Different unit - we classify **sentences**: 12,119 / 1,798 / 3,547.
+ "none" - 9,027 of 12,119 train sentences (74.5%) have no hallmark.
+ Rare labels in dev - replicative immortality 11 sentences, immune destruction 20, growth suppressors 23.

## 5. MASSIVE

**Citation.** FitzGerald et al. "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages." arXiv 2204.08582. Code: https://github.com/alexa/massive .

### Data

+ Scope - 51 languages, 18 domains, 60 intents, 55 slot types (abstract).
+ Origin - professional translators localised the English SLURP data into 50 languages.
+ Size - 19,521 utterances per language (Table 1). Over all locales: 587k train, 104k dev, 152k test, 153k held out for MMNLU-22 (section 1).
+ fr-FR is one of the 51 locales (Table 2).
+ Our release is MASSIVE 1.1: 11,514 / 2,033 / 2,974 per locale (sources note section 6 and 7). The paper describes 1.0.

### Baseline models (section 5.1)

+ XLM-R Base (270M) - JointBERT style: one intent head on the pooled output, one slot head on the token outputs.
+ mT5 Base Encoder-Only (258M) - the same two heads on the mT5 encoder.
+ mT5 Base Text-to-Text (580M) - input "Annotate:" plus the utterance. Output is one label per token, then the intent.
+ Tuning - 128 trials per model (TPE plus ASHA). Checkpoint chosen by exact match over all locales.
+ "Full" setting - training on the full training set of all locales. "Zero-shot" setting - training on en-US only, test on the other locales.

### Metrics

+ Intent accuracy.
+ Slot F1 - micro-averaged, span level. The repo code converts token labels to BIO and calls `seqeval.metrics.f1_score` (`src/massive/utils/training_utils.py`).
+ Exact match accuracy - intent correct and every slot tag correct for the utterance.

### Results

Locale average, full training (Table 3a): intent 85.3 / 86.1 / 85.1, slot F1 76.8 / 75.4 / 73.6, exact match 66.6 / 65.9 / 63.7 for mT5 T2T / mT5 Enc / XLM-R.

Per locale (Tables 7, 8, 9):

| Locale | Setting | Model | Slot F1 | Intent acc | Exact match |
|---|---|---|---|---|---|
| en-US | full | mT5 T2T | 81.6 | 87.9 | 72.5 |
| en-US | full | mT5 Enc | 80.4 | 89.0 | 72.0 |
| en-US | full | XLM-R | 78.7 | 88.3 | 69.7 |
| fr-FR | full | mT5 T2T | 75.6 | 86.9 | 66.2 |
| fr-FR | full | mT5 Enc | 73.5 | 87.2 | 65.1 |
| fr-FR | full | XLM-R | 70.9 | 86.3 | 62.2 |
| fr-FR | zero-shot (en-US train) | mT5 T2T | 54.2 | 76.9 | 47.2 |
| fr-FR | zero-shot (en-US train) | mT5 Enc | 51.2 | 74.1 | 39.5 |
| fr-FR | zero-shot (en-US train) | XLM-R | 59.1 | 80.8 | 48.6 |

+ 95% intervals are about plus or minus 0.5 to 0.7 for slot F1 and 1.2 to 1.7 for the other two metrics.
+ Zero-shot exact match is 25 to 37 points below full training (5.2).

### Limitations

+ The "full" baselines train on all 51 locales together, not on one locale.
+ The paper reports translation and localisation noise.
+ Slot F1 uses word-level BIO tags with seqeval. It depends on tokenisation and spacing.

## 6. Training pitfalls for multi-label classification with GLiNER-style heads

### Class imbalance

+ Problem - most label decisions are negative. In our Hallmarks train set, 74.5% of sentences have no label. The rarest label, cellular energetics, has 136 positive sentences of 12,119.
+ Evidence - the CNN paper says imbalance gives "high precision, low recall, and thus comparatively low F-scores". Oversampling gave +0.8 dev F (Baker, Korhonen, Pyysalo 2016, 4.1).
+ Our loss - `gliner2` uses unweighted BCE for classification. Each sentence adds ten binary terms, so negatives dominate the gradient.
+ Options in the literature - oversample positives (CNN paper); focal loss with alpha 0.7 (GLiClass Table 1). Neither source gives an ablation on a GLiNER head for this.

### Rare labels

+ Micro-F1 hides rare labels, because frequent labels carry most of the counts.
+ Our dev has 11 positives for replicative immortality and 20 for immune destruction. One sentence changes that label's recall by 5 to 9 points.
+ Report macro-F1 and per-label F1 with the micro score. The design already names macro-F1 as secondary.

### Thresholds

+ GLiNER2 tutorial - 0.3 to 0.5 for multi-label.
+ GLiClass lists "calibration variability across datasets" as a limitation.
+ CNN paper - F and AUC can rank models in a different order because of where the decision boundary sits.
+ Our code - one global threshold from the grid {0.3, 0.4, 0.5, 0.6, 0.7}, chosen on dev (`run.py`, `choose_eval_threshold`).
+ Per-label thresholds are common practice, but no source here tests them. With 11 dev positives, a per-label threshold is noisy.

### Label descriptions

+ GLiNER2 tutorial - descriptions "significantly" improve accuracy. No numbers.
+ GLiClass `train.py` adds random descriptions in 5% of training examples.
+ Our student - `tasks/classification.py` passes only the ten natural label names. `configs/label_definitions.yaml` has one definition per hallmark, but only the teacher prompt v2 uses them, and the teacher gate kept v1 for Hallmarks.

### Number of labels

+ GLiClass - attention to label tokens weakens as label count grows. Banking77 (77 labels) is a weak case (2.3.3, section 4).
+ GLiNER2 - Banking77 0.70 vs SNIPS 0.83 (Table 2).
+ Our Hallmarks has 10 labels, so label count is not the main risk. Our MASSIVE slot schema has 55 fields in one structure (`tasks/slots.py`). This is closer to the risky range for short utterances ("many labels and short texts").

### Augmentation

+ GLiClass `train.py` turns on label shuffling and random label removal, addition, synonyms, descriptions and examples by default.
+ GLiClass post-training varies positive and negative label density per text.
+ Our frozen recipes have `augmentation: false` (`configs/recipes/classification.yaml`, `slots.yaml`).

### Sentence versus abstract context

+ The expert labelled only sentences with "findings or conclusions". A sentence with the same topic but a background role gets no label.
+ A sentence-only input cannot see its role in the abstract. This is my reading of the annotation rule, not a tested result.

### LoRA capacity

+ GLiClass used LoRA rank 384 to 1536 and saw more stable training with higher ranks.
+ The GLiNER2 tutorial says to raise the rank to 32 or 64 and add task heads if LoRA is weak.
+ Our frozen classification recipe: r 32, alpha 64, targets encoder plus classification head, `task_lr` 4.0e-4.

## 7. Reference scores for our datasets

Our scores are **not directly comparable** with any published number below. The table says why.

| Dataset | Source | Setting | Metric | Score | Why not comparable |
|---|---|---|---|---|---|
| Hallmarks | BLURB Table 6 | PubMedBERT, full fine-tune, abstract level | micro-F1, 10 labels | 82.32 | abstracts, not sentences |
| Hallmarks | BLURB Table 6 | BERT uncased (general domain) | micro-F1 | 80.20 | abstracts |
| Hallmarks | BLUE Table 3 | BERT-Large PubMed | example-based F1, abstract | 87.3 | other split, no-hallmark abstracts removed |
| Hallmarks | CNN paper Table 4 | tuned CNN | mean per-label binary F | 81.0 | other split, abstracts |
| Hallmarks | Baker 2016 Table 4 | SVM, 4-fold CV | mean per-label F | 76.9 | 1,499 abstracts, CV |
| Hallmarks | any | sentence level, BLURB split | micro-F1 | not found | no published sentence-level score found |
| MASSIVE en-US | MASSIVE Table 9 | XLM-R Base, all-locale training | seqeval slot F1 (test) | 78.7 | multilingual training, BIO span F1, v1.0 |
| MASSIVE en-US | MASSIVE Table 9 | mT5 Enc / mT5 T2T | seqeval slot F1 | 80.4 / 81.6 | same |
| MASSIVE en-US | MASSIVE Table 7 | XLM-R / mT5 Enc / mT5 T2T | exact match | 69.7 / 72.0 / 72.5 | needs the intent |
| MASSIVE fr-FR | MASSIVE Table 9 | XLM-R / mT5 Enc / mT5 T2T, all-locale training | seqeval slot F1 | 70.9 / 73.5 / 75.6 | same |
| MASSIVE fr-FR | MASSIVE Table 9 | en-US training only, zero-shot | seqeval slot F1 | 59.1 / 51.2 / 54.2 | same |
| MASSIVE intent | GLiClass Table 3 | gliclass-large-v3.0, zero-shot | F1 | 0.5649 | intent, not slots |
| MASSIVE intent | GLiNER2 Table 2 | GLiNER2, zero-shot, 31 labels | accuracy | 0.53 | intent, not slots |

Gap between locales in the same paper, full training: en-US minus fr-FR slot F1 is 6.0 (mT5 T2T), 6.9 (mT5 Enc), 7.8 (XLM-R).

## 8. What it means for us

Our student is GLiNER2.5-multi with LoRA. After 20 dev-only tuning trials it reaches dev micro-F1 0.441 on Hallmarks and a dev score of 0.730 on MASSIVE en-US slots (commits `8cb8d8d`, `8bd4bde`).

### Hallmarks: 0.44 is low, but the reference is not like for like

+ Different task - published numbers near 80 to 82 micro-F1 are abstract level. An abstract gets the union of its sentence labels, so one clear sentence is enough. We score each sentence, and 74.5% of sentences are "none". This task is harder, and I found no published sentence-level number on this split.
+ Loss - our loss has no answer to the 10-to-1 imbalance. GLiClass uses focal alpha 0.7. The CNN paper oversamples positives.
+ Threshold - one global threshold and no calibration. GLiClass names calibration as a known weak point.
+ No descriptions - the student gets bare label names. The GLiNER2 authors recommend descriptions.
+ Dev noise - dev has 527 positive label instances in total, and 11 for the rarest label. Small recipe differences can be noise.

Candidate changes. Each one changes the frozen recipe or the design, so Codex and Abhishek must agree first (AGENTS.md section 1):

1. Add a dev-only check of one fixed-recipe run with the label definitions as descriptions.
2. Add class-balanced loss (focal alpha or a positive weight) as a tuning dimension, if `gliner2` exposes one. The current code does not.
3. Report macro-F1 and per-label F1 next to micro-F1, and the full-pool ground-truth reference, so readers see the ceiling.
4. Consider per-label thresholds only if dev counts allow it. With 11 positives they do not.
5. Say in the paper that the Hallmarks task is sentence level and not comparable with BLURB.

For the research question the absolute score matters less than the gap between arms. But a low ceiling compresses the gaps between arms. The full-pool reference run tells us how much room the arms have.

### MASSIVE: 0.73 is close to published fine-tuned baselines

+ Published en-US slot F1 with full supervised training is 78.7 to 81.6 on test. Those models train on all 51 locales (about 50 times more utterances) and use seqeval BIO F1.
+ Our 0.730 is dev, en-US only, and canonical (slot, value) micro-F1. The design already says "Scores are not comparable with published MASSIVE results."
+ So our student is in a reasonable range. The literature gives no strong reason to change the slot recipe.
+ Schema size - the student sees all 55 slot types as fields of one structure, without descriptions. GLiClass reports that many labels with short texts degrade the text vectors. MASSIVE utterances are short. Field descriptions from `configs/label_definitions.yaml` are a cheap dev check.

### MASSIVE fr-FR: expect a drop

+ With full supervised training, the MASSIVE paper shows fr-FR slot F1 6.0 to 7.8 points below en-US.
+ Our fr-FR arm trains on French pool labels (Gemma or ground truth), so the "full" gap is the closer guide, not the zero-shot gap.
+ Expect a lower fr-FR score than en-US. A drop of about 6 to 8 points would match the literature, but no source tests GLiNER2.5-multi on French slots.
+ French slot noise - the MASSIVE authors report localisation noise. Our sources note records 433 fr-FR ids with slot types different from en-US.

## Sources

+ GLiClass paper - https://arxiv.org/abs/2508.07662
+ GLiClass repo - https://github.com/Knowledgator/GLiClass (`README.md`, `train.py`, `gliclass/loss_functions.py`)
+ GLiClass model card - https://huggingface.co/knowledgator/gliclass-large-v3.0
+ GLiNER2 paper - https://arxiv.org/abs/2507.18546
+ GLiNER2 tutorials - https://github.com/fastino-ai/GLiNER2/tree/main/tutorial (files 1, 3, 8, 9, 14)
+ GLiNER2.5 multi model card - https://huggingface.co/fastino/gliner2.5-multi-v1
+ `gliner2` 2.0.0 source in `.venv/lib/python3.11/site-packages/gliner2/`
+ GLiNER multi-task - https://arxiv.org/abs/2406.12925
+ Zero-shot slot filling with GLiNER - https://arxiv.org/abs/2411.18980
+ Baker et al. 2016 - https://academic.oup.com/bioinformatics/article/32/3/432/1743783
+ Baker, Korhonen, Pyysalo 2016 - https://aclanthology.org/W16-5101.pdf
+ BLUE - https://arxiv.org/abs/1906.05474
+ BLURB - https://arxiv.org/abs/2007.15779
+ BigBio Hallmarks card - https://huggingface.co/datasets/bigbio/hallmarks_of_cancer
+ MASSIVE - https://arxiv.org/abs/2204.08582 and https://github.com/alexa/massive

## Changelog

- 2026-09-25 23:25 CEST - Created from a full read of the papers.
