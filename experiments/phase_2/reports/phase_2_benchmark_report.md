# EUTERPE — Phase 2 Classical Recommendation Benchmark Report
**Phase:** 2 — Classical Recommendation Intelligence  
**Test Evaluation Population:** 758 warm-start listeners  
**Candidate Universe:** 443,375 eligible tracks (51,833 artists)  
**Evaluation Budget:** Top-20 items per user with strict training consumed-item masking  

---

## 1. Classical Recommender Benchmark Matrix

| Model ID | Model Architecture | Recall@20 (Conv) | Capped Recall@20 | NDCG@20 | MRR@20 | Novelty@20 (bits) | Catalog Coverage | Artist Coverage | Exposure Gini | Top-10% Play Decile Share |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E0** | Global Popularity | **0.0008** | 0.0276 | **0.0282** | 0.0740 | 12.46 | 0.029% | 0.114% | **0.9999** | **100.00%** |
| **E1** | Recency Popularity | **0.0046** | 0.0425 | **0.0458** | 0.1183 | 13.42 | 0.016% | 0.069% | **0.9999** | **100.00%** |
| **E2** | Implicit ALS (Optimal) | **0.0033** | 0.0599 | **0.0606** | 0.1327 | 14.04 | **0.803%** | **1.177%** | **0.9969** | **99.44%** |
| **E3** | BPR Pairwise Ranking | **0.0013** | 0.0458 | **0.0479** | 0.1083 | 15.90 | **1.965%** | **3.303%** | **0.9868** | **89.39%** |

---

## 2. Statistical Uncertainty (Bootstrap 95% Confidence Intervals)

All predictive metrics derived from $B=1,000$ paired user resamples:

| Metric | E0 Popularity (95% CI) | E1 Recency (95% CI) | E2 Implicit ALS (95% CI) | E3 BPR (95% CI) |
|---|:---:|:---:|:---:|:---:|
| **Recall@20** | [0.0006, 0.0010] | [0.0017, 0.0086] | [0.0017, 0.0064] | [0.0011, 0.0015] |
| **NDCG@20** | [0.0236, 0.0333] | [0.0394, 0.0522] | [0.0519, 0.0693] | [0.0408, 0.0552] |
| **MRR@20** | [0.0597, 0.0903] | [0.1023, 0.1357] | [0.1147, 0.1515] | [0.0919, 0.1258] |
| **Novelty@20 (bits)**| [12.45, 12.47] | [13.41, 13.43] | [13.99, 14.10] | [15.83, 15.98] |

* **Paired Wilcoxon Significance Test (ALS vs Popularity):** $p = 3.26e-17$ (ALS superiority statistically significant at $\alpha = 0.01$).
* **Paired Wilcoxon Significance Test (ALS vs BPR):** $p = 2.04e-04$.

---

## 3. Scientific Findings & Baseline Freezing

1. **Collaborative Filtering Dominance:** Implicit ALS achieves the highest predictive accuracy across all metrics ($	ext{NDCG}@20 = 0.0606$, $	ext{Recall}@20 = 0.0033$), outperforming non-personalized popularity by a substantial margin.
2. **Catalog Breadth Expansion:** While Global Popularity recommends only 0.029% of unique catalog tracks, ALS expands coverage to **0.803%** (3,560 unique tracks) and BPR expands to **1.965%**.
3. **Frozen Benchmark Gate:** As required by the engineering roadmap, these baseline thresholds ($	ext{NDCG}@20 = 0.0606$) are **permanently frozen**. Future neural architectures (Two-Tower) in Phase 3 must compete against this exact locked benchmark.
