# EUTERPE — Sequential Recommendation Benchmark Report
**Document ID:** `EUTERPE-PH8-SEQ-001`  
**Focus:** Benchmarking Sequential Models (SASRec & GRU4Rec) vs. Classical Implicit ALS  
**Evaluation:** Frozen Academic Test Partition (758 Warm Users, $K=20$, 443,375 Items)  
**Safeguards:** Zero future leakage; consumed training items masked (`scores[consumed] = -inf`)  

---

### Executive Summary: Sequential-Hybrid Beats Implicit ALS on Test NDCG@20 (Pure SASRec Trails Baseline)

In Phase 3 and Phase 4, two-tower neural retrieval expanded catalog and artist diversity (reducing top-decile exposure from 99.44% to 75.27%), but fell short of Implicit ALS on raw ranking accuracy:
- **Phase 3 Two-Tower NDCG@20:** `0.0458` vs. **ALS:** `0.0606` (Accuracy target missed).
- **Core Diagnosis:** Static average pooling over multi-year user scrobble histories discarded session transitions, chronological trajectory, and acoustic momentum.

To determine whether sequential modeling can surpass collaborative filtering accuracy on a 443,375-item catalog, EUTERPE evaluated state-of-the-art **Sequential Recommendation Architectures**:
1. **SASRec (Self-Attentive Sequential Recommendation):** Multi-head causal self-attention over chronological scrobble sequences.
2. **GRU4Rec (Gated Recurrent Unit):** Multi-layer recurrent session-based transition modeling.
3. **Sequential-Hybrid Engine (SASRec + ALS):** Dynamic sequential attention blended with collaborative filtering factor representations.

#### Key Empirical Results:
- **Pure SASRec (NDCG@20 = 0.0597):** Trails Implicit ALS (`0.0606`, $-1.4\%$, $p = 0.1357$). Pure self-attention trained from scratch without collaborative factor initialization does not beat ALS on raw ranking accuracy.
- **GRU4Rec (NDCG@20 = 0.0611):** Achieves a marginal $+0.8\%$ lift on NDCG, but exhibits lower recall (`0.0018` vs. `0.0033`).
- **Sequential-Hybrid SASRec + ALS (NDCG@20 = 0.0653):** Achieves a **+7.7% relative lift over Implicit ALS** ($p = 0.02755$, Wilcoxon signed-rank test), with Recall@20 of `0.0033`.
- **Verdict:** **Only the Sequential-Hybrid definitively beats Implicit ALS with statistical significance.**

---

### 1. Benchmark Matrix: Sequential Models vs. Frozen Classical Baseline

| Model Architecture | Recall@20 | Capped Recall@20 | NDCG@20 | MRR@20 | Novelty (bits) | Catalog Coverage | Artist Coverage | Exposure Gini | Top-10% Share | Accuracy vs. ALS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Implicit ALS (Frozen Baseline)** | 0.0033 | 0.0599 | **0.0606** | 0.1327 | 14.04 | 0.803% | 1.177% | 0.9969 | 99.44% | Baseline |
| **Two-Tower (Phase 3 N5)** | 0.0012 | 0.0413 | 0.0458 | 0.1136 | **15.95** | **2.119%** | **3.706%** | **0.9855** | **75.27%** | -24.4% (Missed) |
| **SASRec (Self-Attentive)** | 0.0018 | 0.0567 | 0.0597 | 0.1286 | 13.96 | 0.764% | 1.063% | 0.9971 | 99.40% | -1.4% ($p = 0.1357$) |
| **GRU4Rec (Recurrent Gated)** | 0.0018 | 0.0569 | 0.0611 | 0.1342 | 13.99 | 0.780% | 1.105% | 0.9970 | 99.39% | +0.8% (Marginal) |
| **Sequential-Hybrid (SASRec + ALS)** | **0.0033** | **0.0637** | **0.0653** | **0.1417** | 13.97 | 0.774% | 1.152% | 0.9971 | 99.47% | **+7.7% ($p = 0.0275$)** |

---

### 2. Statistical Significance & Hypothesis Verification

- **Pure SASRec vs. Implicit ALS:**
  - $\Delta \text{NDCG@20}$: **-0.0009** ($-1.4\%$ relative lift over ALS)
  - Paired Wilcoxon Signed-Rank Test: $W = 5512.0, p = 0.1357$ (Fail to reject $H_0$; does not beat ALS)
  - $\Delta \text{Recall@20}$: **-0.0015** ($-46.0\%$ relative lift)
  - **Scientific Interpretation:** Training sequential self-attention on raw sparse scrobbles without collaborative factor pretraining fails to capture global user-item affinity across a 443k item long tail.

- **Sequential-Hybrid (SASRec + ALS) vs. Implicit ALS:**
  - $\Delta \text{NDCG@20}$: **+0.0047** ($+7.7\%$ relative lift over ALS)
  - Paired Wilcoxon Signed-Rank Test: $W = 6780.0, p = 0.02755$ (Statistically significant at $\alpha = 0.05$)
  - $\Delta \text{Recall@20}$: **+0.0000** ($+0.5\%$ relative lift)
  - **Scientific Interpretation:** Blending dynamic chronological attention scores with collaborative latent factors leverages both recent session intent and global long-term preference.

---

### 3. Scientific Analysis: Accuracy vs. Exposure Trade-offs

1. **Why Pure SASRec Needed Collaborative Hybridization:**
   Music catalogs exhibit extreme scale ($443,375$ items in this academic benchmark). Pure sequence encoders trained via contrastive cross-entropy over local sequences suffer from sample sparsity in negative pools, failing to optimize the long tail of items. Hybridizing sequential attention scores with ALS collaborative factors grounds recent temporal signals in robust low-rank representations.

2. **Catalog Exposure Realities (Head vs. Tail):**
   - Sequential models trained directly on scrobble likelihood (SASRec: $99.40\%$ top-decile exposure, Hybrid: $99.47\%$) mirror the popularity concentration of Implicit ALS ($99.44\%$).
   - Sequential modeling alone **does not solve the discovery / head-bias problem**.
   - Expanding tail coverage (as demonstrated by Two-Tower N5 at $75.27\%$ top-decile share and $+164\%$ catalog coverage) requires dedicated discovery mechanisms: log-$Q$ debiased contrastive learning or Phase 4 multi-objective reranking with novelty surprisal.

---

### 4. Production Integration

The serialized Sequential-Hybrid pipeline combines:
1. **Candidate Retrieval:** Blended SASRec sequence scoring (`experiments/phase_8/models/sasrec_model.pt`) and ALS latent factor projection.
2. **Discovery Reranking:** Multi-objective Pareto reranking (Phase 4) with novelty surprisal and artist diversity constraints ($C=2$).
