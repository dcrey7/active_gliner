# Research: Active Learning for NER - Literature Review

**Date:** 2026-04-01 16:21

---

## 1. What is the Standard/Proper Active Learning Loop?

### Formal Definition

Active learning (AL) is formally defined as an **iterative, human-in-the-loop framework** where a learning algorithm interactively selects the most informative unlabeled examples for annotation, retrains on the augmented labeled set, and repeats (Settles, 2009). The canonical pool-based active learning loop is:

1. Train model on initial labeled seed set L
2. Use the model to score all examples in unlabeled pool U with an acquisition function
3. Select the top-k most informative examples
4. Have an oracle (human) label them
5. Add newly labeled examples to L
6. **Retrain the model on the updated L**
7. Repeat from step 2 until budget is exhausted or performance target is met

### Iterative Re-ranking vs. Single-Round Ranking

The **iterative re-ranking** (multi-round) approach is what the literature considers "true" active learning. The critical element is that **the model is retrained between selection rounds**, so the uncertainty estimates are updated based on what the model has already learned. This allows the model to shift its attention to different regions of the data space as it improves.

**Single-round ranking** -- where you score all data once with a fixed model, rank by uncertainty, and train on the top-N -- is more accurately described as **"uncertainty-based data selection"** or **"uncertainty-based prioritization"**. While it uses AL acquisition functions, it lacks the iterative model-update loop that defines active learning.

### Is Single-Round Selection "Active Learning"?

Strictly speaking, **no** -- a single round of ranking + training is not active learning in the formal sense. It is better characterized as:

- **"Uncertainty-based data selection"** or **"confidence-based data prioritization"**
- A single iteration of pool-based active learning
- **"Curriculum learning by uncertainty"** (selecting training data ordered by difficulty/uncertainty)

However, there is a pragmatic gray area. Many applied papers (especially in NLP) perform what amounts to single-round or few-round selection due to computational constraints of retraining deep models. The Shen et al. (2018) ICLR paper on deep AL for NER specifically notes that retraining deep models at every AL round is expensive, which motivated their lightweight CNN-CNN-LSTM architecture that supports efficient incremental training.

**Key distinction:** The literature draws a line between using uncertainty scores for **static data selection** (one-time ranking) versus using them within an **iterative feedback loop** where the model's improving knowledge drives progressively better selection decisions.

### What Your Codebase Does

Looking at `exp_active_learning_confidence_strategies.py`, the current implementation:
- Scores all training data once with a pre-trained GLiNER model
- Ranks by various strategies (MSE, min, MNLP, random)
- Trains on the top-N for various N values
- Does NOT retrain the scoring model between selection rounds

This is **single-round uncertainty-based data selection**, not iterative active learning. This is still a valid and useful experiment (comparing acquisition functions for data prioritization), but should be framed carefully in writing.

---

## 2. Established Acquisition Functions for NER Active Learning

### Core Papers

#### Shen et al. (2017/2018) - "Deep Active Learning for Named Entity Recognition" (ICLR 2018)

- **Architecture:** CNN-CNN-LSTM (convolutional character + word encoders, LSTM tag decoder) -- lightweight for iterative retraining
- **Acquisition functions tested:**
  - **Least Confidence (LC):** Select sentences where the model's best prediction sequence has the lowest probability. Problem: disproportionately selects longer sentences.
  - **Maximum Normalized Log Probability (MNLP):** MNLP = (-1/n) * sum(log(p_i)). Length-normalized version of LC. Faster than BALD (no multiple forward passes needed). **This is the paper that introduced MNLP.**
  - **Bayesian Active Learning by Disagreement (BALD):** Uses MC Dropout to estimate epistemic uncertainty via prediction disagreement across stochastic forward passes.
- **Key result:** MNLP and BALD outperform LC, especially in early AL rounds. Reached near-SOTA with only 25% of training data.
- **Setup:** Proper iterative AL loop with incremental model updates between rounds.

#### Siddhant & Lipton (2018) - "Deep Bayesian Active Learning for NLP" (EMNLP 2018)

- **Large-scale empirical study** across multiple NLP tasks including NER and semantic role labeling.
- **Acquisition functions tested:**
  - **BALD (Bayesian Active Learning by Disagreement):** Uses MC Dropout; measures mutual information between predictions and model parameters. The fraction of dropout models that disagree with the most popular prediction.
  - **MNLP:** Maximum normalized log probability.
  - **Variation Ratios**
  - **Random baseline**
- **Key result:** BALD exhibits state-of-the-art performance across tasks. Bayesian approaches (using dropout or Bayes-by-Backprop for uncertainty) significantly outperform i.i.d. baselines.

### Complete Taxonomy of NER Active Learning Acquisition Functions

| Function | Type | Formula/Description | Reference |
|----------|------|---------------------|-----------|
| **Least Confidence (LC)** | Uncertainty | U(x) = 1 - P(y*|x), select lowest-confidence predictions | Settles (2009) |
| **Margin Sampling** | Uncertainty | U(x) = P(y1|x) - P(y2|x), select smallest margin between top-2 | Settles (2009) |
| **Entropy** | Uncertainty | H = -sum(P(y|x) log P(y|x)), select highest entropy | Settles (2009) |
| **MNLP** | Uncertainty | (-1/n) * sum(log(p_i)), length-normalized log prob | Shen et al. (2018) |
| **BALD** | Bayesian | Mutual information via MC Dropout disagreement | Gal et al. (2017), Siddhant & Lipton (2018) |
| **Expected Gradient Length (EGL)** | Gradient-based | Select samples that would cause largest gradient update | Settles et al. (2008) |
| **Query by Committee (QBC)** | Committee-based | Disagreement among ensemble of models | Seung et al. (1992) |
| **Token-level LC / TLC** | NER-specific | Aggregate token-level least confidence to sentence level | Various |
| **Modified LC for NER** | NER-specific | Weighted LC accounting for number of uncertain tokens | Patra & Chakraborty (2021) |

### How These Map to Sequence Tagging vs. Span Models

The **critical challenge for NER AL** is that acquisition functions designed for classification (one prediction per instance) must be adapted for structured outputs (multiple entities per sentence). Traditional approaches:

- **Token-level:** Compute uncertainty per BIO tag, aggregate to sentence level (sum, mean, max)
- **Sequence-level (MNLP):** Use the probability of the entire tag sequence from CRF/Viterbi decoding
- **BALD for sequences:** MC Dropout across the full sequence, measure disagreement at token level

---

## 3. Active Learning for Span-Based NER Models

### Has This Been Done? -- Likely Novel

After extensive searching, I found **no published work** that applies active learning specifically to span-based NER models (like GLiNER, SpERT, S-NER, T2-NER). All existing NER active learning papers work with **sequence taggers** (BiLSTM-CRF, BERT+CRF, CNN-LSTM+CRF with BIO/BIOES tagging).

### Why This Matters

Span-based models like GLiNER fundamentally change how uncertainty is computed:

- **Sequence taggers:** Uncertainty comes from token-level tag probabilities or CRF sequence scores. MNLP operates on the Viterbi path probability. BALD uses MC Dropout on token-level predictions.
- **Span-based models (GLiNER):** Uncertainty comes from **span-entity matching scores** -- the similarity between a span representation and an entity type embedding in a shared latent space. Each candidate span gets a confidence score independently.

This creates a **different uncertainty landscape:**
- No sequence-level probability to normalize (no CRF/Viterbi)
- Each entity prediction is an independent span with its own score
- A sentence's uncertainty is naturally an **aggregation of per-entity span scores**
- The "no entities predicted" case (empty prediction) has a distinct meaning

### Novelty Assessment

**Applying active learning to span-based NER models appears to be novel.** The combination of:
1. Span-level (not token-level) confidence scores
2. Acquisition functions adapted for variable numbers of predicted spans per sentence
3. Application to GLiNER's span-entity matching paradigm

...does not appear in published literature as of early 2026. This is a genuine contribution worth highlighting.

---

## 4. MSE as an Acquisition Function

### What Your Code Implements

From `strategy.py`:
```python
# MSE = mean of (1.0 - score)^2 for all predicted entities
squared_errors = [(1.0 - score) ** 2 for score in entity_scores]
mse = sum(squared_errors) / len(squared_errors)
```

This measures the average squared distance of entity confidence scores from perfect confidence (1.0). Higher MSE = more uncertain.

### Is This Novel?

**Partially.** MSE itself is not novel as a concept, but its specific application as an acquisition function for NER data selection is uncommon:

- **What exists in literature:** MSE has been used as a **learning objective** (loss function) in active learning, and as an **evaluation metric** for regression-based AL. Some work uses secondary models to estimate prediction MSE as an acquisition signal. The broader concept of "distance from ideal confidence" underlies least confidence (which uses 1 - max_prob, the L1 distance from 1 for one prediction).
- **What appears novel:** Using MSE specifically as `mean((1 - confidence_i)^2)` aggregated across multiple predicted spans in a span-based NER model. This is essentially a **quadratic penalty version of least confidence**, aggregated across variable-length entity predictions.

### How MSE Relates to Established Functions

| Your Function | Closest Standard Equivalent | Key Difference |
|---------------|---------------------------|----------------|
| **MSE:** mean((1-s_i)^2) | Least Confidence: 1 - max(s_i) | MSE penalizes low scores quadratically; averages across all entities instead of taking worst/max |
| **Min score:** min(s_i) | Least Confidence (adapted) | Directly equivalent to entity-level least confidence |
| **Avg score:** mean(s_i) | Mean Confidence | Standard confidence averaging |
| **MNLP:** -mean(log(s_i)) | MNLP (Shen et al. 2018) | Direct implementation of the standard MNLP, adapted for span scores |

**MSE's unique property:** The quadratic penalty means it is more sensitive to very low confidence scores than linear alternatives. A score of 0.3 contributes 0.49 to MSE but only 0.7 to a linear penalty. This makes MSE more aggressive at detecting "very uncertain" entities while being more forgiving of "slightly uncertain" ones.

### Recommendation for Framing

Do NOT claim MSE is "novel" outright. Instead frame it as: "We adapt MSE as an uncertainty aggregation function for span-based predictions, where it serves as a quadratic generalization of least confidence applied to variable-length entity sets." This is honest and still highlights the contribution.

---

## 5. Key Criticisms of Single-Round Active Learning Simulations

### Margatina & Aletras (2023) - "On the Limitations of Simulating Active Learning" (ACL Findings 2023)

**Core argument:** Most AL research uses simulated experiments on pre-labeled datasets, which introduces systematic biases:

- **The simulation assumption:** That labels exist for the entire pool, so the "oracle" just reveals pre-existing labels. This ignores real annotation noise, inter-annotator disagreement, and the cost structure of actual annotation.
- **Cold start / warm-up problem:** AL algorithms have unpredictable warm-up times -- a minimum number of labeled instances before gains over random sampling appear. Simulated settings often mask this.
- **Setting governs findings:** The specific simulation parameters (seed set size, batch size, number of rounds, dataset) can dramatically change which AL method "wins." Results don't generalize across settings.
- **Lower bound argument:** Simulated AL experiments provide only a lower bound on real effectiveness, missing practical insights like negative results and obstacles.

### Lowell, Lipton & Wallace (2019) - "Practical Obstacles to Deploying Active Learning" (EMNLP 2019)

**Core findings:**
- **Model coupling:** AL couples the training dataset to the specific model used for acquisition. An actively-acquired dataset does NOT consistently outperform i.i.d. sampled data when used to train a **different successor model.**
- **Lack of generalization:** Benefits of AL do not generalize reliably across models and tasks.
- **The fixed-model problem:** If you use Model A to select data, then train Model B on that data, Model B may not benefit (or may even be harmed) because the selection was optimized for Model A's blind spots.

### Specific Criticisms Relevant to Single-Round Approaches

1. **Homogeneity problem:** Similar examples produce similar prediction probabilities. A single-round query selects a homogeneous batch. Iterative rounds allow the model to "move past" already-learned patterns and diversify its selections.

2. **No feedback correction:** Without retraining between rounds, the scoring model cannot correct its own errors. Early mistakes in uncertainty estimation propagate to the entire selected set.

3. **Sampling bias:** Single-round selection induces systematic bias toward certain data characteristics (e.g., longer sentences for LC). Iterative rounds with retraining can partially correct this bias.

4. **Batch diversity:** Single-round methods that select top-k by uncertainty tend to pick redundant examples. Iterative methods naturally introduce diversity because the retrained model becomes confident on regions already well-covered.

5. **Calibration issues:** Deep learning models produce poorly calibrated confidence scores (softmax outputs are not true probabilities). A single round amplifies miscalibration effects; iterative rounds allow the model to self-correct as it sees more data.

### How This Applies to Your Work

Your current experiment (`exp_active_learning_confidence_strategies.py`) performs **single-round data selection** (rank all data with base GLiNER, train on top-N). The criticisms above apply:

- The ranking is done once with the pre-trained model and never updated
- Top-N selected examples may be redundant (homogeneity)
- The experiment still has value as a comparison of acquisition functions for **data prioritization**, but should not be called "active learning" without qualification

**Mitigation options:**
- Frame as "uncertainty-based data selection" or "acquisition function comparison for data prioritization"
- Add an iterative version where the model is retrained every K examples and re-ranks the remaining pool
- Compare single-round vs. iterative to quantify the benefit of re-ranking

---

## 6. Recent Work (2023-2025): Key Papers

### "Have LLMs Made Active Learning Obsolete?" (2025 Community Survey)

- Conducted an NLP community survey on AL adoption
- **Answer: No.** Data annotation remains important, AL stays relevant while benefiting from LLMs
- Persistent challenges: setup complexity, uncertain cost reduction, tooling -- same challenges as 15 years ago
- LLMs are being integrated into AL pipelines (for annotation, as backbone models) but haven't replaced the AL paradigm
- GPU-accelerated language modeling plays a key role; trends center on integrating small LLMs into AL components

### Active Learning for Clinical NER (2024)

- Evaluated AL strategies with BioBERT for clinical NER
- Proposed diversity-based (CLUSTER via Sentence-BERT) and hybrid strategies (CLC, CNBSE) that switch from diversity to uncertainty
- Fewer AL iterations needed vs. random selection

### Transformer Uncertainty for AL (2024)

- Compared 8 alternative uncertainty measures for Transformer models
- Found that softmax probabilities are misleading for uncertainty
- Most uncertainty methods primarily identify hard-to-learn-from samples rather than samples that actually reduce model uncertainty
- Suggests need for better-calibrated uncertainty for AL to work well

### On the Fragility of Active Learners for Text Classification (2024)

- Shows that AL benefits are fragile and sensitive to hyperparameters
- Reinforces the Lowell et al. (2019) findings about lack of robustness

---

## 7. Summary and Recommendations for Your Project

### What You Can Claim

1. **Novelty in applying AL-style acquisition functions to span-based NER (GLiNER)** -- this appears genuinely new. All prior NER AL work uses sequence taggers.
2. **MSE as a quadratic uncertainty aggregation** -- novel in this specific context, but should be framed carefully as an adaptation, not a breakthrough.
3. **Comparison of acquisition functions for span-based NER data prioritization** -- valuable contribution even without iterative retraining.

### What to Be Careful About

1. Do not call single-round selection "active learning" without qualification. Use "uncertainty-based data selection" or "single-round AL simulation."
2. Acknowledge the limitations of single-round approaches (homogeneity, no feedback correction, calibration).
3. If comparing to random baseline, note that random selection often provides better diversity, which can offset the benefit of uncertainty-based selection.

### What Would Strengthen the Work

1. **Add iterative retraining:** Even 3-5 rounds of retrain-and-rerank would make this true active learning.
2. **Compare single-round vs. iterative** to quantify the gap.
3. **Add a diversity component:** Combine uncertainty with clustering/diversity (e.g., k-means on sentence embeddings) to address the homogeneity problem.
4. **Calibration analysis:** Show whether GLiNER's span scores are well-calibrated, since this directly affects acquisition function quality.

---

## Sources

- [Shen et al. 2018 - Deep Active Learning for NER (ICLR)](https://openreview.net/pdf?id=ry018WZAZ)
- [Siddhant & Lipton 2018 - Deep Bayesian Active Learning for NLP (EMNLP)](https://aclanthology.org/D18-1318/)
- [Settles 2009 - Active Learning Literature Survey](https://burrsettles.com/pub/settles.activelearning.pdf)
- [Margatina & Aletras 2023 - On the Limitations of Simulating Active Learning (ACL Findings)](https://aclanthology.org/2023.findings-acl.269/)
- [Lowell, Lipton & Wallace 2019 - Practical Obstacles to Deploying Active Learning (EMNLP)](https://aclanthology.org/D19-1003/)
- [2025 Community Survey - Have LLMs Made Active Learning Obsolete?](https://arxiv.org/abs/2503.09701)
- [GLiNER - Generalist Model for NER (NAACL 2024)](https://aclanthology.org/2024.naacl-long.300/)
- [Gal et al. 2017 - Deep Bayesian Active Learning with Image Data](https://arxiv.org/abs/1703.02910)
- [Patra & Chakraborty 2021 - Modified LC for NER](https://link.springer.com/article/10.1007/s13748-021-00230-w)
- [Lilian Weng 2022 - Active Learning Overview](https://lilianweng.github.io/posts/2022-02-20-active-learning/)
- [Comparing Uncertainty Measures for Transformer Models (2024)](https://link.springer.com/article/10.1007/s10796-024-10503-z)
- [Active Learning for Clinical NER (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11491619/)
